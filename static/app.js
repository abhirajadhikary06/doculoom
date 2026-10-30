/* DocuLoom — Campus Knowledge Intelligence Engine
   Frontend: neobrutalism chat + admin panels + provenance explorer */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  // ── API helper ──────────────────────────────────────────────────────────────
  const api = (path, opts = {}) => {
    const headers = { ...(opts.headers || {}) };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    if (tenant) headers["X-Tenant-Id"] = tenant;
    if (opts.body && !(opts.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.body);
    }
    return fetch(path, { ...opts, headers }).then(async (r) => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw Object.assign(new Error(data.error || r.statusText), { status: r.status, data });
      return data;
    });
  };

  // ── State ───────────────────────────────────────────────────────────────────
  let token = localStorage.getItem("dl_token") || "";
  let user = JSON.parse(localStorage.getItem("dl_user") || "null");
  let tenant = localStorage.getItem("dl_tenant") || "campus";
  let conversationId = null;
  let authMode = "login";
  let useV2 = localStorage.getItem("dl_v2") === "true";

  // ── Theme ───────────────────────────────────────────────────────────────────
  const theme = localStorage.getItem("dl_theme") || "light";
  document.documentElement.setAttribute("data-theme", theme);
  $("#themeBtn").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("dl_theme", next);
  });

  // ── V2 toggle ───────────────────────────────────────────────────────────────
  $("#v2Toggle").checked = useV2;
  $("#v2Toggle").addEventListener("change", () => {
    useV2 = $("#v2Toggle").checked;
    localStorage.setItem("dl_v2", useV2);
  });

  // ── User UI ─────────────────────────────────────────────────────────────────
  function setUserUI() {
    $("#userLabel").textContent = user ? user.display_name || user.email : "guest";
    $("#tenantLabel").textContent = user?.tenant_id || tenant;
    $("#authBtn").textContent = user ? "Sign out" : "Sign in";
  }
  setUserUI();

  // ── Auth modal ──────────────────────────────────────────────────────────────
  const authModal = $("#authModal");
  $("#authBtn").addEventListener("click", () => {
    if (user) {
      token = ""; user = null;
      localStorage.removeItem("dl_token");
      localStorage.removeItem("dl_user");
      setUserUI();
      return;
    }
    authModal.classList.remove("hidden");
  });
  $("#closeAuth").addEventListener("click", () => authModal.classList.add("hidden"));
  $("#toggleAuthMode").addEventListener("click", () => {
    authMode = authMode === "login" ? "register" : "login";
    $("#authTitle").textContent = authMode === "login" ? "Sign in" : "Create account";
    $("#authSubmit").textContent = authMode === "login" ? "Sign in" : "Register";
    $("#toggleAuthMode").textContent = authMode === "login" ? "Need an account?" : "Have an account?";
    $("#nameLabel").classList.toggle("hidden", authMode === "login");
  });
  $("#authForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = $("#authEmail").value.trim();
    const password = $("#authPassword").value;
    const display_name = $("#authName").value.trim();
    try {
      const path = authMode === "login" ? "/api/auth/login" : "/api/auth/register";
      const body = { email, password };
      if (authMode === "register") body.display_name = display_name || email.split("@")[0];
      const data = await api(path, { method: "POST", body });
      token = data.token; user = data.user; tenant = user.tenant_id;
      localStorage.setItem("dl_token", token);
      localStorage.setItem("dl_user", JSON.stringify(user));
      localStorage.setItem("dl_tenant", tenant);
      setUserUI();
      authModal.classList.add("hidden");
      refreshDocs();
    } catch (err) { alert(err.message || "Auth failed"); }
  });

  // ── Documents ───────────────────────────────────────────────────────────────
  async function refreshDocs() {
    try {
      const data = await api("/api/docs");
      const list = $("#docList");
      list.innerHTML = "";
      (data.documents || []).forEach((d) => {
        const li = document.createElement("li");
        const trustPct = d.trust_score != null ? Math.round(d.trust_score * 100) + "%" : "";
        li.innerHTML = `<span class="name" title="${d.filename}">${d.filename}</span>
          <span class="badge ${d.status}">${d.status} · ${d.chunk_count || 0}c${trustPct ? " · " + trustPct : ""}</span>`;
        list.appendChild(li);
      });
    } catch (e) { console.warn(e); }
  }
  refreshDocs();

  // ── Upload ───────────────────────────────────────────────────────────────────
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault(); dropzone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); });

  async function uploadFile(file) {
    const status = $("#uploadStatus");
    status.className = "status"; status.textContent = `Uploading ${file.name}…`;
    const fd = new FormData();
    fd.append("file", file);
    try {
      const data = await api("/api/ingest", { method: "POST", body: fd });
      if (data.status === "ready") {
        status.className = "status ok";
        status.textContent = `✓ ${data.filename} · ${data.page_count} pages · ${data.chunk_count} chunks`;
      } else {
        status.className = "status err";
        status.textContent = `Failed: ${data.error || data.status}`;
      }
      refreshDocs();
    } catch (err) {
      status.className = "status err"; status.textContent = err.message || "Upload failed";
    }
  }

  // ── Chat ─────────────────────────────────────────────────────────────────────
  const messagesEl = $("#messages");

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function renderMarkdown(text) {
    return escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br/>")
      .replace(/\[([^\]]+),\s*page\s*(\d+)(?:-(\d+))?\]/gi, (_, fn, p, pe) =>
        `<span class="cite">[${escapeHtml(fn)}, page ${p}${pe ? "-" + pe : ""}]</span>`
      );
  }

  function addMessage(role, content, meta = {}) {
    const { sources, latency, verification, strategy, intent, difficulty,
            wasDecomposed, subQuestions, cacheHit, logId, temporalWarning } = meta;
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = `<p>${renderMarkdown(content)}</p>`;

    // Temporal warning
    if (temporalWarning) {
      const tw = document.createElement("div");
      tw.style = "margin-top:6px;font-size:0.78rem;color:#e88a00;font-style:italic";
      tw.textContent = "⚠ " + temporalWarning;
      bubble.appendChild(tw);
    }

    // Intelligence metadata badges (v2)
    if (strategy || difficulty || cacheHit || wasDecomposed) {
      const badges = document.createElement("div");
      badges.className = "intel-meta";
      if (strategy) badges.innerHTML += `<span class="intel-badge strategy">${strategy}</span>`;
      if (difficulty) badges.innerHTML += `<span class="intel-badge">${difficulty}</span>`;
      if (cacheHit) badges.innerHTML += `<span class="intel-badge cached">cache:${cacheHit}</span>`;
      if (wasDecomposed) badges.innerHTML += `<span class="intel-badge decomposed">decomposed</span>`;
      if (subQuestions && subQuestions.length > 1) {
        subQuestions.forEach((sq, i) => {
          badges.innerHTML += `<span class="intel-badge" title="${escapeHtml(sq)}">sub-q ${i + 1}</span>`;
        });
      }
      bubble.appendChild(badges);
    }

    // Evidence verification
    if (verification && verification.total_claims > 0) {
      const ev = document.createElement("div");
      ev.style = "margin-top:8px;font-size:0.78rem;color:var(--muted)";
      const cov = Math.round((verification.citation_coverage || 0) * 100);
      const col = cov >= 80 ? "var(--ok)" : cov >= 50 ? "#e88a00" : "var(--accent)";
      ev.innerHTML = `<span style="color:${col}">✓ ${verification.supported_claims}/${verification.total_claims} claims verified · ${cov}% coverage</span>`;
      bubble.appendChild(ev);
    }

    // Sources
    if (sources && sources.length) {
      const src = document.createElement("div");
      src.className = "sources";
      src.innerHTML = "<h4>Sources</h4>";
      sources.forEach((s) => {
        const chip = document.createElement("span");
        chip.className = "source-chip";
        chip.title = s.preview || "";
        let label = `${s.filename} · p.${s.page}${s.page_end && s.page_end !== s.page ? "-" + s.page_end : ""}`;
        if (s.trust_score != null) label += ` [T:${Math.round(s.trust_score * 100)}%]`;
        chip.textContent = label;
        src.appendChild(chip);
      });
      if (latency != null) {
        const lat = document.createElement("div");
        lat.style = "margin-top:6px;color:var(--muted);font-size:0.75rem";
        lat.textContent = `${Math.round(latency)} ms`;
        src.appendChild(lat);
      }
      // Provenance link
      if (logId) {
        const link = document.createElement("button");
        link.className = "btn btn-ghost btn-xs";
        link.style = "margin-top:6px";
        link.textContent = "🔍 View trace";
        link.addEventListener("click", () => showProvenance(logId));
        src.appendChild(link);
      }
      bubble.appendChild(src);
    }

    div.appendChild(bubble);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ── Ask form ─────────────────────────────────────────────────────────────────
  $("#askForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#question").value.trim();
    if (!q) return;
    $("#question").value = "";
    addMessage("user", q);
    const btn = $("#askBtn");
    btn.disabled = true; btn.textContent = "…";
    const thinking = document.createElement("div");
    thinking.className = "msg assistant";
    thinking.innerHTML = `<div class="bubble typing">Thinking</div>`;
    messagesEl.appendChild(thinking);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
      const endpoint = useV2 ? "/api/ask/v2" : "/api/ask";
      const data = await api(endpoint, {
        method: "POST",
        body: { question: q, conversation_id: conversationId },
      });
      thinking.remove();
      if (data.conversation_id) conversationId = data.conversation_id;
      addMessage("assistant", data.answer, {
        sources: data.sources,
        latency: data.latency_ms,
        verification: data.verification,
        strategy: data.retrieval_strategy,
        intent: data.query_intent,
        difficulty: data.difficulty,
        wasDecomposed: data.was_decomposed,
        subQuestions: data.sub_questions,
        cacheHit: data.cache_hit || null,
        logId: data.log_id,
      });
      $("#latLabel").textContent = `${Math.round(data.latency_ms)} ms`;
    } catch (err) {
      thinking.remove();
      addMessage("assistant", `Error: ${err.message || "request failed"}`);
    } finally {
      btn.disabled = false; btn.textContent = "Ask";
    }
  });

  $("#question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#askForm").requestSubmit(); }
  });

  // ── Admin Panel ───────────────────────────────────────────────────────────────
  const adminPanel = $("#adminPanel");
  $("#adminBtn").addEventListener("click", () => {
    adminPanel.classList.remove("hidden");
    loadDashboard();
  });
  $("#closeAdmin").addEventListener("click", () => adminPanel.classList.add("hidden"));

  // Tab switching
  $$(".nav-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".nav-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const tabId = tab.dataset.tab;
      // Hide all tab content
      $$("[id^='tab-']").forEach((el) => el.classList.add("hidden"));
      $(`#tab-${tabId}`).classList.remove("hidden");
    });
  });

  // ── Dashboard ───────────────────────────────────────────────────────────────
  async function loadDashboard() {
    try {
      const data = await api("/api/admin/dashboard");
      // Doc health
      const docStats = data.documents || {};
      const statGrid = $("#docHealthStats");
      statGrid.innerHTML = "";
      const statusColors = { ready: "ok", failed: "crit", quarantined: "warn", processing: "" };
      for (const [st, cnt] of Object.entries(docStats)) {
        statGrid.innerHTML += `<div class="stat-box ${statusColors[st] || ''}">
          <div class="val">${cnt}</div><div class="label">${st}</div></div>`;
      }

      // Gap summary
      const gs = data.knowledge_gaps || {};
      $("#gapSummaryWidget").innerHTML = `
        <div class="stat-grid">
          <div class="stat-box"><div class="val">${gs.total_gaps || 0}</div><div class="label">Gap clusters</div></div>
          <div class="stat-box warn"><div class="val">${gs.total_unanswered_questions || 0}</div><div class="label">Unanswered</div></div>
        </div>`;

      // Conflicts
      const cs = data.conflicts || {};
      const critClass = cs.unresolved > 0 ? "crit" : "ok";
      $("#conflictSummaryWidget").innerHTML = `
        <div class="stat-grid">
          <div class="stat-box ${critClass}"><div class="val">${cs.unresolved || 0}</div><div class="label">Unresolved</div></div>
        </div>
        <div style="margin-top:8px">
        ${(cs.recent || []).slice(0, 2).map((c) =>
          `<div style="font-size:0.78rem;margin-bottom:4px">
            <span class="sev sev-${c.severity}">${c.severity}</span> ${escapeHtml((c.description || "").substring(0, 80))}
          </div>`).join("")}
        </div>`;

      // Cache
      const cache = data.cache || {};
      $("#cacheMetricsWidget").innerHTML = `
        <div class="stat-grid">
          <div class="stat-box"><div class="val">${pct(cache.exact_hit_rate)}</div><div class="label">Exact hits</div></div>
          <div class="stat-box"><div class="val">${pct(cache.semantic_hit_rate)}</div><div class="label">Semantic hits</div></div>
          <div class="stat-box"><div class="val">${cache.active_entries || 0}</div><div class="label">Entries</div></div>
        </div>`;

      // Drift
      const dr = data.drift;
      if (dr) {
        const alertClass = dr.alert_triggered ? "crit" : "ok";
        $("#driftSummaryWidget").innerHTML = `
          <div class="stat-grid">
            <div class="stat-box ${alertClass}"><div class="val">${pct(dr.recall_at_5)}</div><div class="label">Recall@5</div></div>
            <div class="stat-box"><div class="val">${dr.drift_pct_display || "—"}</div><div class="label">Drift</div></div>
          </div>
          ${dr.alert_triggered ? '<div style="color:var(--accent);font-size:0.8rem;margin-top:4px">⚠ Drift alert triggered</div>' : ''}`;
      } else {
        $("#driftSummaryWidget").innerHTML = '<p style="color:var(--muted);font-size:0.8rem">No drift runs yet</p>';
      }

      // Expiry
      const exp = data.expiry || {};
      $("#expirySummaryWidget").innerHTML = `
        <div class="stat-grid">
          <div class="stat-box crit"><div class="val">${exp.expired || 0}</div><div class="label">Expired</div></div>
          <div class="stat-box warn"><div class="val">${exp.superseded || 0}</div><div class="label">Superseded</div></div>
          <div class="stat-box"><div class="val">${exp.pending_review || 0}</div><div class="label">Review Due</div></div>
        </div>`;
    } catch (e) { console.warn("Dashboard load error:", e); }
  }

  function pct(v) { return v != null ? Math.round(v * 100) + "%" : "—"; }

  // Cache invalidate
  $("#invalidateCacheBtn").addEventListener("click", async () => {
    try {
      const r = await api("/api/cache/invalidate", { method: "POST" });
      alert(`Invalidated ${r.invalidated} cache entries`);
    } catch (e) { alert("Error: " + e.message); }
  });

  // Run drift
  $("#runDriftBtn").addEventListener("click", async () => {
    $("#runDriftBtn").disabled = true; $("#runDriftBtn").textContent = "Running…";
    try {
      const r = await api("/api/drift/evaluate", { method: "POST" });
      if (r.error) {
        alert("Drift error: " + r.error);
      } else {
        alert(`Drift run complete. Recall@${r.k}: ${pct(r[`recall_at_${r.k}`])} · Drift: ${r.drift_pct_display}`);
        loadDashboard();
      }
    } catch (e) { alert("Drift error: " + e.message); }
    finally { $("#runDriftBtn").disabled = false; $("#runDriftBtn").textContent = "Run Drift Eval"; }
  });

  // Refresh expiry
  $("#refreshExpiryBtn").addEventListener("click", async () => {
    try {
      const r = await api("/api/expiry/refresh", { method: "POST" });
      alert("Expiry statuses refreshed: " + JSON.stringify(r.status_counts));
      loadDashboard();
    } catch (e) { alert("Error: " + e.message); }
  });

  // ── Conflicts tab ────────────────────────────────────────────────────────────
  $("#loadConflictsBtn").addEventListener("click", loadConflicts);
  async function loadConflicts() {
    const sev = $("#conflictSeverityFilter").value;
    const url = "/api/conflicts" + (sev ? `?severity=${sev}` : "");
    try {
      const data = await api(url);
      const tbody = $("#conflictsBody");
      tbody.innerHTML = "";
      if (!data.conflicts.length) {
        tbody.innerHTML = '<tr><td colspan="7" style="color:var(--muted)">No conflicts found</td></tr>';
        return;
      }
      data.conflicts.forEach((c) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code style="font-size:0.75rem">${c.conflict_type}</code></td>
          <td><span class="sev sev-${c.severity}">${c.severity}</span></td>
          <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">${escapeHtml((c.claim_a || "").substring(0, 100))}</td>
          <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">${escapeHtml((c.claim_b || "").substring(0, 100))}</td>
          <td style="font-size:0.75rem">${escapeHtml(c.filename_a || "")} / ${escapeHtml(c.filename_b || "")}</td>
          <td>${c.is_temporal ? "✓" : ""}</td>
          <td><button class="btn btn-xs" data-id="${c.id}">Resolve</button></td>`;
        tr.querySelector("[data-id]").addEventListener("click", async (e) => {
          const cid = e.target.dataset.id;
          await api(`/api/conflicts/${cid}/resolve`, { method: "POST" });
          loadConflicts();
        });
        tbody.appendChild(tr);
      });
    } catch (e) { console.warn(e); }
  }

  // ── Expiry tab ───────────────────────────────────────────────────────────────
  $("#loadExpiryBtn").addEventListener("click", loadExpiry);
  async function loadExpiry() {
    const st = $("#expiryStatusFilter").value;
    const url = "/api/expiry" + (st ? `?status=${st}` : "");
    try {
      const data = await api(url);
      const tbody = $("#expiryBody");
      tbody.innerHTML = "";
      if (!data.report.length) {
        tbody.innerHTML = '<tr><td colspan="5" style="color:var(--muted)">No records</td></tr>';
        return;
      }
      data.report.forEach((r) => {
        const tr = document.createElement("tr");
        const tagClass = { active: "tag-active", expired: "tag-expired",
          superseded: "tag-superseded", pending_review: "tag-review", unknown: "tag-unknown" }[r.expiry_status] || "";
        tr.innerHTML = `
          <td>${escapeHtml(r.filename || r.document_id)}</td>
          <td><span class="tag-badge ${tagClass}">${r.expiry_status}</span></td>
          <td>${r.expiry_date || "—"}</td>
          <td>${r.review_date || "—"}</td>
          <td style="font-size:0.75rem;color:var(--muted)">${r.expiry_source || ""}</td>`;
        tbody.appendChild(tr);
      });
    } catch (e) { console.warn(e); }
  }

  // ── Quality tab ──────────────────────────────────────────────────────────────
  $("#loadQualityBtn").addEventListener("click", loadQuality);
  async function loadQuality() {
    try {
      const data = await api("/api/quality");
      const tbody = $("#qualityBody");
      tbody.innerHTML = "";
      if (!data.reports.length) {
        tbody.innerHTML = '<tr><td colspan="7" style="color:var(--muted)">No reports</td></tr>';
        return;
      }
      data.reports.forEach((r) => {
        const score = r.quality_score || 0;
        const qClass = score >= 0.8 ? "good" : score >= 0.5 ? "fair" : "poor";
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${escapeHtml(r.filename || r.document_id)}</td>
          <td><span class="quality-score ${qClass}">${(score * 100).toFixed(0)}%</span></td>
          <td>${r.text_coverage != null ? (r.text_coverage * 100).toFixed(0) + "%" : "—"}</td>
          <td>${r.zero_text_pages || 0} / ${r.total_pages || 0} pages</td>
          <td>${r.empty_chunks || 0}</td>
          <td>${r.quarantined ? '<span style="color:var(--accent)">⚠ Yes</span>' : "No"}</td>
          <td>${r.quarantined
            ? `<button class="btn btn-xs btn-primary" data-id="${r.document_id}">Reprocess</button>`
            : "—"}</td>`;
        const btn = tr.querySelector("[data-id]");
        if (btn) {
          btn.addEventListener("click", async (e) => {
            btn.disabled = true; btn.textContent = "…";
            try {
              const res = await api(`/api/quality/${e.target.dataset.id}/reprocess`, { method: "POST" });
              alert("Reprocess: " + (res.status || JSON.stringify(res)));
              loadQuality();
            } catch (ex) { alert("Error: " + ex.message); }
            finally { btn.disabled = false; btn.textContent = "Reprocess"; }
          });
        }
        tbody.appendChild(tr);
      });
    } catch (e) { console.warn(e); }
  }

  // ── Gaps tab ─────────────────────────────────────────────────────────────────
  $("#loadGapsBtn").addEventListener("click", loadGaps);
  $("#aggregateGapsBtn").addEventListener("click", async () => {
    try {
      await api("/api/gaps/aggregate", { method: "POST" });
      loadGaps();
    } catch (e) { console.warn(e); }
  });
  async function loadGaps() {
    try {
      const data = await api("/api/gaps?min_count=1");
      const container = $("#gapsList");
      container.innerHTML = "";
      if (!data.gaps.length) {
        container.innerHTML = '<p style="color:var(--muted);font-size:0.85rem">No knowledge gaps detected yet.</p>';
        return;
      }
      data.gaps.forEach((g) => {
        const card = document.createElement("div");
        card.className = "intel-card";
        card.style = "margin-bottom:10px;padding:12px";
        const samples = (g.sample_questions || []).slice(0, 3)
          .map((q) => `<li style="font-size:0.8rem">${escapeHtml(q)}</li>`).join("");
        card.innerHTML = `
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <strong>${escapeHtml(g.cluster_label || "Unknown gap")}</strong>
              <span class="gap-pill">${g.category || "general"}</span>
            </div>
            <div style="text-align:right">
              <span class="stat-box" style="display:inline-block;padding:4px 10px">
                <span class="val" style="font-size:1rem">${g.question_count}</span>
                <span class="label">asks</span>
              </span>
            </div>
          </div>
          <ul style="margin-top:8px;padding-left:16px">${samples}</ul>
          <p style="font-size:0.78rem;color:var(--accent-2);margin-top:8px">${escapeHtml(g.recommendation || "")}</p>
          <button class="btn btn-xs" data-id="${g.gap_id}" style="margin-top:8px">Mark resolved</button>`;
        card.querySelector("[data-id]").addEventListener("click", async (e) => {
          await api(`/api/gaps/${e.target.dataset.id}/resolve`, { method: "POST" });
          loadGaps();
        });
        container.appendChild(card);
      });
    } catch (e) { console.warn(e); }
  }

  // ── Trust tab ────────────────────────────────────────────────────────────────
  $("#loadTrustBtn").addEventListener("click", loadTrust);
  $("#recomputeTrustBtn").addEventListener("click", async () => {
    try {
      const r = await api("/api/trust/recompute", { method: "POST" });
      alert(`Recomputed trust for ${r.updated} documents`);
      loadTrust();
    } catch (e) { alert("Error: " + e.message); }
  });
  async function loadTrust() {
    try {
      const data = await api("/api/trust");
      const tbody = $("#trustBody");
      tbody.innerHTML = "";
      if (!data.trust_scores.length) {
        tbody.innerHTML = '<tr><td colspan="5" style="color:var(--muted)">No scores</td></tr>';
        return;
      }
      data.trust_scores.forEach((t) => {
        const score = t.trust_score || 0;
        const fillClass = score >= 0.7 ? "" : score >= 0.4 ? "mid" : "low";
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${escapeHtml(t.filename || t.document_id)}</td>
          <td>
            <div class="trust-bar">
              <div class="trust-track"><div class="trust-fill ${fillClass}" style="width:${(score * 100).toFixed(0)}%"></div></div>
              <span style="font-family:var(--mono);font-size:0.78rem">${(score * 100).toFixed(0)}%</span>
            </div>
          </td>
          <td style="font-size:0.78rem;color:var(--muted)">${escapeHtml(t.interpretation || "")}</td>
          <td>${t.superseded ? '<span style="color:var(--accent)">Yes</span>' : "No"}</td>
          <td style="font-family:var(--mono);font-size:0.78rem">${t.pub_date || "—"}</td>`;
        tbody.appendChild(tr);
      });
    } catch (e) { console.warn(e); }
  }

  // ── Drift tab ────────────────────────────────────────────────────────────────
  $("#addBaselineBtn").addEventListener("click", async () => {
    const question = $("#baselineQuestion").value.trim();
    if (!question) return;
    const kws = $("#baselineKeywords").value.trim().split(",").map((k) => k.trim()).filter(Boolean);
    try {
      await api("/api/drift/baseline", { method: "POST", body: { question, expected_keywords: kws } });
      $("#baselineStatus").textContent = "✓ Baseline question added";
      $("#baselineQuestion").value = "";
      $("#baselineKeywords").value = "";
    } catch (e) { $("#baselineStatus").textContent = "Error: " + e.message; }
  });

  $("#loadDriftBtn").addEventListener("click", loadDriftHistory);
  async function loadDriftHistory() {
    try {
      const data = await api("/api/drift/history");
      const container = $("#driftHistory");
      container.innerHTML = "";
      if (!data.history.length) {
        container.innerHTML = '<p style="color:var(--muted);font-size:0.8rem">No drift runs recorded yet. Add baseline questions then run an evaluation.</p>';
        return;
      }
      const maxRecall = Math.max(...data.history.map((r) => r.recall_at_5 || 0), 0.01);
      const barsEl = document.createElement("div");
      barsEl.className = "drift-history";
      data.history.slice().reverse().forEach((r) => {
        const h = Math.max(4, Math.round((r.recall_at_5 / maxRecall) * 44));
        const bar = document.createElement("div");
        bar.className = `drift-bar${r.alert_triggered ? " alert" : ""}`;
        bar.style.height = h + "px";
        bar.title = `${r.created_at ? r.created_at.substring(0, 10) : ""}: Recall=${pct(r.recall_at_5)} Drift=${r.drift_pct_display}`;
        barsEl.appendChild(bar);
      });
      container.appendChild(barsEl);
      const latest = data.history[0];
      container.innerHTML += `<p style="font-size:0.78rem;margin-top:8px">Latest: Recall@5 ${pct(latest.recall_at_5)} · MRR ${pct(latest.mrr)} · Drift ${latest.drift_pct_display}
        ${latest.alert_triggered ? ' <span style="color:var(--accent)">⚠ Alert</span>' : ''}</p>`;
    } catch (e) { console.warn(e); }
  }

  // ── Provenance tab ───────────────────────────────────────────────────────────
  $("#loadProvenanceBtn").addEventListener("click", loadProvenance);
  async function loadProvenance() {
    try {
      const data = await api("/api/provenance?limit=20");
      const tbody = $("#provenanceBody");
      tbody.innerHTML = "";
      if (!data.logs.length) {
        tbody.innerHTML = '<tr><td colspan="7" style="color:var(--muted)">No logs yet. Use /api/ask/v2 to generate provenance records.</td></tr>';
        return;
      }
      data.logs.forEach((l) => {
        const tr = document.createElement("tr");
        const dt = l.created_at ? new Date(l.created_at).toLocaleString() : "—";
        tr.innerHTML = `
          <td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escapeHtml(l.question)}">${escapeHtml(l.question)}</td>
          <td><span class="intel-badge strategy">${l.retrieval_strategy || "—"}</span></td>
          <td style="font-size:0.75rem">${l.difficulty_class || "—"}</td>
          <td style="font-family:var(--mono);font-size:0.75rem">${l.latency_ms ? Math.round(l.latency_ms) + "ms" : "—"}</td>
          <td>${l.cache_hit ? "✓" : ""}</td>
          <td style="font-size:0.75rem;color:var(--muted)">${dt}</td>
          <td><button class="btn btn-xs" data-id="${l.log_id}">Trace</button></td>`;
        tr.querySelector("[data-id]").addEventListener("click", (e) => showProvenance(e.target.dataset.id));
        tbody.appendChild(tr);
      });
    } catch (e) { console.warn(e); }
  }

  // ── Provenance modal ─────────────────────────────────────────────────────────
  async function showProvenance(logId) {
    const modal = $("#provenanceModal");
    const content = $("#provenanceContent");
    content.innerHTML = '<div class="loading-dots" style="color:var(--muted)">Loading trace</div>';
    modal.classList.remove("hidden");
    try {
      const trace = await api(`/api/provenance/${logId}`);
      content.innerHTML = renderProvenance(trace);
    } catch (e) {
      content.innerHTML = `<p style="color:var(--accent)">Error loading trace: ${e.message}</p>`;
    }
  }

  function renderProvenance(trace) {
    const steps = [
      { label: "User query", detail: escapeHtml(trace.question) },
      { label: "Query classification", detail: `Intent: <strong>${trace.query_intent || "—"}</strong> · Difficulty: <strong>${trace.difficulty_class || "—"}</strong>` },
      { label: "Search strategy", detail: `Strategy: <strong>${trace.retrieval_strategy || "—"}</strong>` },
      { label: "Embedding model", detail: `<code>${trace.embedding_model || "—"}</code>` },
      { label: "Evidence retrieved", detail: `${trace.evidence_count} passages retrieved` },
      { label: "LLM model", detail: `<code>${trace.llm_model || "—"}</code> · Prompt v${trace.prompt_version || "—"}` },
      { label: "Latency", detail: `${trace.latency_ms ? Math.round(trace.latency_ms) + " ms" : "—"}` },
    ];

    const stepsHtml = steps.map((s, i) => `
      <div class="trace-step">
        <div class="trace-num">${i + 1}</div>
        <div class="trace-content">
          <div class="trace-label">${s.label}</div>
          <div class="trace-detail">${s.detail}</div>
        </div>
      </div>`).join("");

    const evidenceHtml = (trace.evidence || []).slice(0, 8).map((ev, i) => `
      <tr>
        <td>${i + 1}</td>
        <td style="font-family:var(--mono);font-size:0.73rem">${escapeHtml(ev.filename || ev.document_id || "—")}</td>
        <td>p.${ev.page || 1}</td>
        <td style="font-family:var(--mono);font-size:0.73rem">${ev.rrf_score != null ? ev.rrf_score.toFixed(4) : "—"}</td>
        <td style="font-family:var(--mono);font-size:0.73rem">${ev.rerank_score != null ? ev.rerank_score.toFixed(4) : "—"}</td>
        <td style="max-width:200px;font-size:0.75rem;overflow:hidden;text-overflow:ellipsis">${escapeHtml((ev.text_snapshot || "").substring(0, 100))}</td>
      </tr>`).join("");

    return `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div>
          <h3 style="font-size:0.78rem;text-transform:uppercase;color:var(--muted);margin-bottom:10px">Execution steps</h3>
          ${stepsHtml}
        </div>
        <div>
          <h3 style="font-size:0.78rem;text-transform:uppercase;color:var(--muted);margin-bottom:10px">Evidence passages</h3>
          <div class="table-wrap">
            <table class="intel-table">
              <thead><tr><th>#</th><th>Document</th><th>Page</th><th>RRF</th><th>Rerank</th><th>Snippet</th></tr></thead>
              <tbody>${evidenceHtml || '<tr><td colspan="6" style="color:var(--muted)">No evidence snapshots</td></tr>'}</tbody>
            </table>
          </div>
          <div style="margin-top:12px">
            <h3 style="font-size:0.78rem;text-transform:uppercase;color:var(--muted);margin-bottom:8px">Answer (hash: ${escapeHtml((trace.answer_hash || "").substring(0, 12))}…)</h3>
            <div style="font-size:0.82rem;background:var(--bg);padding:10px;border-radius:8px;border:1px solid var(--ink)">
              ${escapeHtml((trace.answer || "").substring(0, 400))}${(trace.answer || "").length > 400 ? "…" : ""}
            </div>
          </div>
        </div>
      </div>`;
  }

  $("#closeProvenance").addEventListener("click", () => $("#provenanceModal").classList.add("hidden"));
  $("#provenanceModal").addEventListener("click", (e) => {
    if (e.target === $("#provenanceModal")) $("#provenanceModal").classList.add("hidden");
  });

})();
