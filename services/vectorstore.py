"""Qdrant hybrid (dense + sparse) vector store with RRF fusion."""
from __future__ import annotations
import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import Config

logger = logging.getLogger(__name__)

# FastEmbed models for dense + BM25 sparse
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"


class VectorStore:
    def __init__(self):
        self.collection = Config.QDRANT_COLLECTION
        kwargs = {"url": Config.QDRANT_URL, "prefer_grpc": False}
        if Config.QDRANT_API_KEY:
            kwargs["api_key"] = Config.QDRANT_API_KEY
        try:
            self.client = QdrantClient(**kwargs)
        except Exception as e:
            logger.warning("Qdrant connect failed, falling back to :memory: — %s", e)
            self.client = QdrantClient(location=":memory:")
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            exists = self.client.collection_exists(self.collection)
        except Exception:
            exists = False
        if exists:
            return
        logger.info("Creating Qdrant collection %s (hybrid dense+sparse)", self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=Config.EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                    modifier=models.Modifier.IDF,
                )
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=100,
            ),
        )
        # Payload indexes for multi-tenant filtering & latency
        for field, schema in [
            ("tenant_id", models.PayloadSchemaType.KEYWORD),
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("filename", models.PayloadSchemaType.KEYWORD),
            ("page_start", models.PayloadSchemaType.INTEGER),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        tenant_id: str,
        document_id: str,
        filename: str,
    ) -> int:
        """
        chunks: list of dicts with keys:
          id, text, chunk_index, page_start, page_end, char_start, char_end, token_count
        Uses FastEmbed Document models for automatic dense + sparse embedding.
        """
        if not chunks:
            return 0
        points = []
        for c in chunks:
            pid = c.get("id") or str(uuid.uuid4())
            payload = {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": c["chunk_index"],
                "page_start": c.get("page_start", 1),
                "page_end": c.get("page_end", 1),
                "char_start": c.get("char_start", 0),
                "char_end": c.get("char_end", 0),
                "token_count": c.get("token_count", 0),
                "text": c["text"],
                "text_preview": c["text"][:200],
            }
            points.append(
                models.PointStruct(
                    id=pid,
                    vector={
                        "dense": models.Document(
                            text=c["text"],
                            model=DENSE_MODEL,
                        ),
                        "sparse": models.Document(
                            text=c["text"],
                            model=SPARSE_MODEL,
                        ),
                    },
                    payload=payload,
                )
            )
        # Batch upsert
        batch = 64
        for i in range(0, len(points), batch):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + batch],
                wait=True,
            )
        return len(points)

    def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 8,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Dense + sparse with Reciprocal Rank Fusion (built-in)."""
        must = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            )
        ]
        if document_ids:
            must.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=document_ids),
                )
            )
        query_filter = models.Filter(must=must)

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(text=query, model=DENSE_MODEL),
                        using="dense",
                        limit=max(top_k * 3, 20),
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=models.Document(text=query, model=SPARSE_MODEL),
                        using="sparse",
                        limit=max(top_k * 3, 20),
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            points = response.points
        except Exception as e:
            logger.warning("Hybrid query failed, falling back to dense-only: %s", e)
            # Fallback dense-only
            try:
                response = self.client.query_points(
                    collection_name=self.collection,
                    query=models.Document(text=query, model=DENSE_MODEL),
                    using="dense",
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
                points = response.points
            except Exception as e2:
                logger.error("Dense search also failed: %s", e2)
                return []

        results = []
        for p in points:
            pl = p.payload or {}
            results.append(
                {
                    "id": str(p.id),
                    "score": float(p.score or 0),
                    "text": pl.get("text", ""),
                    "filename": pl.get("filename", ""),
                    "document_id": pl.get("document_id", ""),
                    "page_start": pl.get("page_start", 1),
                    "page_end": pl.get("page_end", 1),
                    "chunk_index": pl.get("chunk_index", 0),
                    "text_preview": pl.get("text_preview", ""),
                }
            )
        return results

    def delete_document(self, document_id: str, tenant_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                        models.FieldCondition(
                            key="tenant_id",
                            match=models.MatchValue(value=tenant_id),
                        ),
                    ]
                )
            ),
        )

    def count(self, tenant_id: Optional[str] = None) -> int:
        if tenant_id:
            return self.client.count(
                collection_name=self.collection,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id",
                            match=models.MatchValue(value=tenant_id),
                        )
                    ]
                ),
                exact=True,
            ).count
        return self.client.count(collection_name=self.collection, exact=True).count
