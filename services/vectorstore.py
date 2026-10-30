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
_st_model = None


class VectorStore:
    def __init__(self):
        self.collection = Config.QDRANT_COLLECTION
        self.is_memory = False
        if Config.QDRANT_URL == ":memory:":
            self.client = QdrantClient(location=":memory:")
            self.is_memory = True
            self._ensure_collection()
            return

        kwargs = {"url": Config.QDRANT_URL, "prefer_grpc": False}
        if Config.QDRANT_API_KEY:
            kwargs["api_key"] = Config.QDRANT_API_KEY
        try:
            self.client = QdrantClient(**kwargs)
            self._ensure_collection()
        except Exception as e:
            logger.warning("Qdrant connect failed, falling back to :memory: — %s", e)
            self.client = QdrantClient(location=":memory:")
            self.is_memory = True
            try:
                self._ensure_collection()
            except Exception as e2:
                logger.warning("In-memory collection setup warning: %s", e2)

    def _embed_dense(self, text: str) -> List[float]:
        global _st_model
        if _st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            except Exception:
                pass
        if _st_model is not None:
            try:
                return _st_model.encode(text).tolist()
            except Exception:
                pass
        import hashlib
        dim = Config.EMBEDDING_DIM
        h = hashlib.sha256(text.encode()).digest()
        vals = [(b / 255.0) - 0.5 for b in h]
        return (vals * ((dim // len(vals)) + 1))[:dim]

    def _embed_sparse(self, text: str) -> models.SparseVector:
        import re, hashlib
        from collections import Counter
        words = re.findall(r"\w+", text.lower())
        if not words:
            return models.SparseVector(indices=[0], values=[0.0])
        counts = Counter(words)
        pairs = sorted(
            [(int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % 100000, float(cnt))
             for w, cnt in counts.items()],
            key=lambda x: x[0]
        )
        indices, values = [], []
        for idx, val in pairs:
            if indices and indices[-1] == idx:
                values[-1] += val
            else:
                indices.append(idx)
                values.append(val)
        return models.SparseVector(indices=indices, values=values)

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
            if self.is_memory:
                vec = {
                    "dense": self._embed_dense(c["text"]),
                    "sparse": self._embed_sparse(c["text"]),
                }
            else:
                vec = {
                    "dense": models.Document(
                        text=c["text"],
                        model=DENSE_MODEL,
                    ),
                    "sparse": models.Document(
                        text=c["text"],
                        model=SPARSE_MODEL,
                    ),
                }
            points.append(
                models.PointStruct(
                    id=pid,
                    vector=vec,
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

        dense_q = self._embed_dense(query) if self.is_memory else models.Document(text=query, model=DENSE_MODEL)
        sparse_q = self._embed_sparse(query) if self.is_memory else models.Document(text=query, model=SPARSE_MODEL)

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_q,
                        using="dense",
                        limit=max(top_k * 3, 20),
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=sparse_q,
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
                    query=dense_q,
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

    def dense_search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 8,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Dense-only (embedding) search — used by adaptive retrieval planner."""
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

        dense_q = self._embed_dense(query) if self.is_memory else models.Document(text=query, model=DENSE_MODEL)
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=dense_q,
                using="dense",
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            points = response.points
        except Exception as e:
            logger.warning("Dense-only search failed: %s", e)
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

    def count_document_chunks(self, document_id: str, tenant_id: str) -> int:
        """Count Qdrant points for a specific document — used by ingestion quality check."""
        try:
            result = self.client.count(
                collection_name=self.collection,
                count_filter=models.Filter(
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
                ),
                exact=True,
            )
            return result.count
        except Exception as e:
            logger.warning("count_document_chunks failed: %s", e)
            return -1

    def get_document_chunks(
        self,
        document_id: str,
        tenant_id: str,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document — used by counterfactual testing."""
        try:
            response = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
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
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            points = response[0]
        except Exception as e:
            logger.warning("get_document_chunks failed: %s", e)
            return []

        results = []
        for p in points:
            pl = p.payload or {}
            results.append(
                {
                    "id": str(p.id),
                    "text": pl.get("text", ""),
                    "filename": pl.get("filename", ""),
                    "document_id": pl.get("document_id", ""),
                    "page_start": pl.get("page_start", 1),
                    "chunk_index": pl.get("chunk_index", 0),
                }
            )
        return sorted(results, key=lambda x: x["chunk_index"])
