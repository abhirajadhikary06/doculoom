/* DocuLoom frontend — neobrutalism chat + ingest */
(function () {
  const $ = (s) => document.querySelector(s);
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

  let token = localStorage.getItem("dl_token") || "";
  let user = JSON.parse(localStorage.getItem("dl_user") || "null");
  let tenant = localStorage.getItem("dl_tenant") || "campus";
  let conversationId = null;
  let authMode = "login"; // login | register

  // Theme
  const theme = localStorage.getItem("dl_theme") || "light";
  document.documentElement.setAttribute("data-theme", theme);
  $("#themeBtn").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("dl_theme", next);
  });

  function setUserUI() {
    $("#userLabel").textContent = user ? user.display_name || user.email : "guest";
    $("#tenantLabel").textContent = user?.tenant_id || tenant;
    $("#authBtn").textContent = user ? "Sign out" : "Sign in";
  }
  setUserUI();

  // Auth modal
  const modal = $("#authModal");
  $("#authBtn").addEventListener("click", () => {
    if (user) {
      token = "";
      user = null;
      localStorage.removeItem("dl_token");
      localStorage.removeItem("dl_user");
      setUserUI();
      return;
    }
    modal.classList.remove("hidden");
  });
  $("#closeAuth").addEventListener("click", () => modal.classList.add("hidden"));
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
      token = data.token;
      user = data.user;
      tenant = user.tenant_id;
      localStorage.setItem("dl_token", token);
      localStorage.setItem("dl_user", JSON.stringify(user));
      localStorage.setItem("dl_tenant", tenant);
      setUserUI();
      modal.classList.add("hidden");
    } catch (err) {
      alert(err.message || "Auth failed");
    }
  });

  // Docs
  async function refreshDocs() {
    try {
      const data = await api("/api/docs");
      const list = $("#docList");
      list.innerHTML = "";
      (data.documents || []).forEach((d) => {
        const li = document.createElement("li");
        li.innerHTML = `<span class="name" title="${d.filename}">${d.filename}</span>
          <span class="badge ${d.status}">${d.status} · ${d.chunk_count || 0}c</span>`;
        list.appendChild(li);
      });
    } catch (e) {
      console.warn(e);
    }
  }
  refreshDocs();

  // Upload
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  });

  async function uploadFile(file) {
    const status = $("#uploadStatus");
    status.className = "status";
    status.textContent = `Uploading ${file.name}…`;
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
      status.className = "status err";
      status.textContent = err.message || "Upload failed";
    }
  }

  // Chat
  const messagesEl = $("#messages");
  function addMessage(role, content, sources, latency) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";

    // Simple markdown-ish: bold, newlines, citations
    let html = escapeHtml(content)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br/>")
      .replace(/\[([^\]]+),\s*page\s*(\d+)(?:-(\d+))?\]/gi, (_, fn, p, pe) => {
        return `<span class="cite">[${escapeHtml(fn)}, page ${p}${pe ? "-" + pe : ""}]</span>`;
      });
    bubble.innerHTML = `<p>${html}</p>`;

    if (sources && sources.length) {
      const src = document.createElement("div");
      src.className = "sources";
      src.innerHTML = "<h4>Sources</h4>";
      sources.forEach((s) => {
        const chip = document.createElement("span");
        chip.className = "source-chip";
        chip.title = s.preview || "";
        chip.textContent = `${s.filename} · p.${s.page}${s.page_end && s.page_end !== s.page ? "-" + s.page_end : ""}`;
        src.appendChild(chip);
      });
      if (latency != null) {
        const lat = document.createElement("div");
        lat.style.marginTop = "6px";
        lat.style.color = "var(--muted)";
        lat.textContent = `${Math.round(latency)} ms`;
        src.appendChild(lat);
      }
      bubble.appendChild(src);
    }
    div.appendChild(bubble);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  $("#askForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#question").value.trim();
    if (!q) return;
    $("#question").value = "";
    addMessage("user", q);
    const btn = $("#askBtn");
    btn.disabled = true;
    btn.textContent = "…";
    const thinking = document.createElement("div");
    thinking.className = "msg assistant";
    thinking.innerHTML = `<div class="bubble typing">Thinking</div>`;
    messagesEl.appendChild(thinking);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
      const data = await api("/api/ask", {
        method: "POST",
        body: { question: q, conversation_id: conversationId },
      });
      thinking.remove();
      if (data.conversation_id) conversationId = data.conversation_id;
      addMessage("assistant", data.answer, data.sources, data.latency_ms);
      $("#latLabel").textContent = `${Math.round(data.latency_ms)} ms`;
    } catch (err) {
      thinking.remove();
      addMessage("assistant", `Error: ${err.message || "request failed"}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Ask";
    }
  });

  // Enter to send (Shift+Enter newline)
  $("#question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("#askForm").requestSubmit();
    }
  });
})();
