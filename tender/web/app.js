// Tender Checker UI (shadcn/ui-style, plain JS, no dependencies).
// SECURITY: tender and bid text is untrusted (a bid is adversarial input). Every piece of document or model
// text is inserted with textContent / createTextNode. Icons are built with DOM calls, never HTML strings.
"use strict";

const $ = (id) => document.getElementById(id);
const STATUSES = ["MISSING", "PARTIAL", "UNCLEAR", "MET"];
const state = { key: "", projects: [], project: null, matrix: null, viewer: null, tab: "overview",
  sort: { key: null, dir: 1 }, lastFocus: null, chats: {} };

// ---------------------------------------------------------------- DOM helpers + icons (Lucide, ISC)

const SVG = "http://www.w3.org/2000/svg";
const ICONS = {
  check: [["circle", { cx: 12, cy: 12, r: 10 }], ["path", { d: "m9 12 2 2 4-4" }]],
  x: [["circle", { cx: 12, cy: 12, r: 10 }], ["path", { d: "m15 9-6 6" }], ["path", { d: "m9 9 6 6" }]],
  alert: [["circle", { cx: 12, cy: 12, r: 10 }], ["path", { d: "M12 8v4" }], ["path", { d: "M12 16h.01" }]],
  help: [["circle", { cx: 12, cy: 12, r: 10 }], ["path", { d: "M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" }], ["path", { d: "M12 17h.01" }]],
  file: [["path", { d: "M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" }], ["path", { d: "M14 2v4a2 2 0 0 0 2 2h4" }],
    ["path", { d: "M16 13H8" }], ["path", { d: "M16 17H8" }], ["path", { d: "M10 9H8" }]],
  folder: [["path", { d: "M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" }]],
  warn: [["path", { d: "m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" }], ["path", { d: "M12 9v4" }], ["path", { d: "M12 17h.01" }]],
  loader: [["path", { d: "M21 12a9 9 0 1 1-6.219-8.56" }]],
  eye: [["path", { d: "M2.06 12.35a1 1 0 0 1 0-.7 10.75 10.75 0 0 1 19.88 0 1 1 0 0 1 0 .7 10.75 10.75 0 0 1-19.88 0" }], ["circle", { cx: 12, cy: 12, r: 3 }]],
  shield: [["path", { d: "M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" }], ["path", { d: "m9 12 2 2 4-4" }]],
  arrowRight: [["path", { d: "M5 12h14" }], ["path", { d: "m12 5 7 7-7 7" }]],
  info: [["circle", { cx: 12, cy: 12, r: 10 }], ["path", { d: "M12 16v-4" }], ["path", { d: "M12 8h.01" }]],
};
const STATUS_ICON = { MET: "check", PARTIAL: "alert", MISSING: "x", UNCLEAR: "help" };

function icon(name, cls = "icon") {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", cls);
  svg.setAttribute("aria-hidden", "true");
  for (const [tag, attrs] of ICONS[name] || []) {
    const n = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    svg.appendChild(n);
  }
  return svg;
}

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    n.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
  }
  return n;
}

function statusBadge(s) { return el("span", { class: `badge badge-${s}` }, icon(STATUS_ICON[s]), s); }
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const fmtUsd = (v) => `$${(v || 0).toFixed(v && v < 0.01 ? 4 : 3)}`;
const fmtDate = (iso) => { try { return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); } catch { return iso; } };

// ---------------------------------------------------------------- toasts

function toast(title, desc = "", kind = "info") {
  const t = el("div", { class: `toast toast-${kind}`, role: kind === "error" ? "alert" : "status" },
    icon(kind === "success" ? "check" : kind === "error" ? "x" : "info"),
    el("div", {}, el("div", { class: "toast-title" }, title), desc ? el("div", { class: "toast-desc" }, desc) : null));
  $("toasts").appendChild(t);
  setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 220); }, kind === "error" ? 7000 : 4000);
}

// ---------------------------------------------------------------- theme

function applyTheme(pref) {
  if (pref === "light" || pref === "dark") document.documentElement.setAttribute("data-theme", pref);
  else document.documentElement.removeAttribute("data-theme");
  document.querySelectorAll("[data-theme-set]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeSet === pref)));
  try { localStorage.setItem("tenderTheme", pref); } catch { /* storage blocked: theme still applies this session */ }
}

// ---------------------------------------------------------------- overlays (dialogs + sheet)

function openDialog(id) {
  state.lastFocus = document.activeElement;
  $("dialogScrim").hidden = false;
  $(id).hidden = false;
  const first = $(id).querySelector("input, button:not([data-close])");
  if (first) setTimeout(() => first.focus(), 10);
}
function closeDialogs() {
  ["keyDialog", "newDialog", "deleteDialog"].forEach((id) => { $(id).hidden = true; });
  $("dialogScrim").hidden = true;
  if (state.lastFocus && state.lastFocus.focus) state.lastFocus.focus();
}
function anyDialogOpen() { return ["keyDialog", "newDialog", "deleteDialog"].some((id) => !$(id).hidden); }

// ---------------------------------------------------------------- API

function loadKey() { try { state.key = localStorage.getItem("tenderKey") || ""; } catch { state.key = ""; } }
function saveKey(k) {
  state.key = k.trim();
  try { localStorage.setItem("tenderKey", state.key); } catch { /* private mode: keep in memory */ }
}

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {}, state.key ? { "X-API-Key": state.key } : {});
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (res.status === 401) { askKey(); throw new Error("unauthorised"); }
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res;
}
const getJSON = async (p) => (await api(p)).json();
const postJSON = async (p, body) => (await api(p, { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}) })).json();

function askKey() { $("keyInput").value = state.key; openDialog("keyDialog"); }

// ---------------------------------------------------------------- boot + projects

async function health() {
  const h = $("health");
  const txt = h.querySelector(".status-text");
  try {
    const r = await (await fetch("/health")).json();
    txt.textContent = r.status === "ok" ? `Ready · ${r.models.pro}` : "Degraded";
    h.className = "status-pill " + (r.status === "ok" ? "ok" : "bad");
    h.title = r.status === "ok" ? `LLM: ${r.llm_backend} (${r.models.flash} / ${r.models.pro})` : "Database or vector store is down";
  } catch { txt.textContent = "Offline"; h.className = "status-pill bad"; }
}

async function boot() {
  health();
  if (!state.key) { $("whoami").textContent = "Add API key"; $("userAvatar").textContent = "?"; renderProjects(); return; }
  try {
    const me = await getJSON("/me");
    $("whoami").textContent = me.name + (me.admin ? " · admin" : "");
    $("userAvatar").textContent = me.name.slice(0, 2).toUpperCase();
    $("budget").value = me.defaults.check_budget_usd.toFixed(2);
    $("budget").max = me.defaults.max_check_budget_usd;
    await loadProjects();
  } catch (e) { if (e.message !== "unauthorised") toast("Could not load your account", e.message, "error"); }
}

async function loadProjects(selectId) {
  state.projects = await getJSON("/projects");
  renderProjects();
  if (selectId) openProject(selectId);
}

function renderProjects() {
  const q = $("projectSearch").value.trim().toLowerCase();
  const list = state.projects.filter((p) => !q || p.name.toLowerCase().includes(q));
  const ul = $("projectList");
  if (!state.key) { ul.replaceChildren(el("li", { class: "project-empty" }, "Add your API key to see projects")); return; }
  if (!list.length) { ul.replaceChildren(el("li", { class: "project-empty" }, q ? "No matching projects" : "No projects yet")); return; }
  ul.replaceChildren(...list.map((p) => {
    const c = p.latest_check;
    const missing = c ? (c.counts.MISSING || 0) : 0;
    const meta = c
      ? [el("span", { class: "mini-dot", style: `background:hsl(var(${missing ? "--missing" : "--met"}))` }),
        missing ? `${missing} missing` : "no gaps", c.counts.PARTIAL ? ` · ${c.counts.PARTIAL} partial` : ""]
      : [plural(p.documents.length, "document")];
    const active = state.project && state.project.project_id === p.project_id;
    return el("li", {}, el("button", { class: "project-item", type: "button", "aria-current": active ? "true" : "false",
      onclick: () => { openProject(p.project_id); closeSidebar(); } }, icon("folder"),
    el("span", { class: "project-text" }, el("span", { class: "project-name", dir: "auto" }, p.name), el("span", { class: "project-meta" }, ...meta))));
  }));
}

async function openProject(id) {
  let p;
  try { p = await getJSON(`/projects/${encodeURIComponent(id)}`); } catch (e) { toast("Could not open the project", e.message, "error"); return; }
  const switched = !state.project || state.project.project_id !== p.project_id;
  state.project = p;
  if (switched) { state.matrix = null; state.sort = { key: null, dir: 1 }; }
  $("empty").hidden = true;
  $("projectView").hidden = false;
  $("projName").textContent = p.name;
  $("crumbProject").textContent = p.name;
  $("projMeta").textContent = `${p.project_id} · created ${fmtDate(p.created_at)} · ${plural(p.checks, "check")}`;
  renderDocs(p);
  renderLastCheck(p);
  renderProjects();
  if (switched) renderChat();
  const running = (p.jobs || []).find((j) => j.status === "queued" || j.status === "running");
  if (running && !state.watching) watchJob(running.job_id, running.kind);
  if (p.latest_check) await loadMatrix(); else { renderNoResults(); }
  if (switched) selectTab(p.latest_check ? state.tab : "overview");
}

function renderDocs(p) {
  const ul = $("docList");
  if (!p.documents.length) {
    ul.replaceChildren(el("li", { class: "doc-empty" }, "No documents yet. Upload the tender and the bid."));
    return;
  }
  ul.replaceChildren(...p.documents.map((d) => el("li", { class: "doc" },
    el("span", { class: "doc-icon" }, icon("file")),
    el("div", { class: "doc-main" },
      el("div", { class: "doc-name", dir: "auto", title: d.filename }, d.filename),
      el("div", { class: "doc-meta" }, el("span", { class: "badge badge-secondary" }, d.role === "tender" ? "Tender" : "Bid"),
        `${plural(d.pages, "page")} · ${plural(d.chunks, "chunk")}`,
        d.hidden_spans ? el("span", { class: "doc-warn", title: "Invisible text was dropped before indexing: a possible prompt-injection attempt" },
          icon("warn", "icon icon-xs"), `${d.hidden_spans} hidden text span(s) dropped`) : null)))));
}

function renderLastCheck(p) {
  const box = $("lastCheck");
  const c = p.latest_check;
  if (!c) { box.hidden = true; return; }
  box.hidden = false;
  box.replaceChildren(
    el("span", {}, "Last check ", el("b", {}, fmtDate(c.created_at))),
    el("span", {}, el("b", {}, c.judged), ` of ${c.requirements} requirements judged`),
    el("span", {}, "cost ", el("b", {}, fmtUsd(c.cost_usd))),
    el("button", { class: "cite", type: "button", onclick: () => selectTab("compliance") }, "View results", icon("arrowRight")));
}

// ---------------------------------------------------------------- tabs

function selectTab(name) {
  state.tab = name;
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === name)));
  ["overview", "compliance", "ask"].forEach((n) => { $(`panel-${n}`).hidden = n !== name; });
  if (name === "ask") setTimeout(() => $("question").focus(), 30);
}

// ---------------------------------------------------------------- upload + jobs

async function upload(role, files) {
  if (!state.project || !files.length) return;
  const fd = new FormData();
  fd.append("role", role);
  for (const f of files) fd.append("files", f);
  try {
    const r = await (await api(`/projects/${state.project.project_id}/documents`, { method: "POST", body: fd })).json();
    for (const x of r.rejected) toast("File rejected", x.error, "error");
    for (const a of r.accepted) await watchJob(a.job_id, "ingest");
  } catch (e) {
    let msg = e.message;
    try { const j = JSON.parse(msg); if (j.rejected) msg = j.rejected.map((x) => x.error).join(" · "); } catch { /* plain text */ }
    toast("Upload failed", msg, "error");
  }
}

function showJob(job) {
  const box = $("jobBox");
  box.hidden = false;
  box.classList.toggle("failed", job.status === "failed");
  box.classList.toggle("indeterminate", job.status !== "done" && job.status !== "failed" && !job.total);
  const label = job.kind === "ingest" ? `${job.params.filename || "file"}: ` : "";
  const stage = $("jobStage");
  const running = job.status === "queued" || job.status === "running";
  stage.replaceChildren(running ? icon("loader", "icon spin") : icon(job.status === "done" ? "check" : "x"),
    job.status === "failed" ? `Failed: ${job.error || "unknown error"}` : label + (job.status === "done" ? "done" : job.stage));
  $("jobCount").textContent = job.total ? `${job.done} / ${job.total}` : "";
  const pct = job.status === "done" ? 100 : job.total ? Math.round(100 * job.done / job.total) : 0;
  $("jobBar").style.width = pct + "%";
}

async function watchJob(jobId, kind) {
  state.watching = true;
  const btn = $("checkBtn");
  btn.disabled = true;
  try {
    for (;;) {
      const job = await getJSON(`/jobs/${jobId}`);
      showJob(job);
      if (job.status === "done" || job.status === "failed") {
        if (job.status === "done" && kind === "check") {
          toast("Check finished", `${job.result.judged} requirements judged · ${fmtUsd(job.result.cost_usd)}`, "success");
        }
        if (job.status === "done" && kind === "ingest") {
          toast(`${job.result.filename} added`, job.result.warnings.length ? job.result.warnings.join(" · ")
            : `${plural(job.result.pages, "page")}, ${plural(job.result.chunks, "chunk")}`, job.result.warnings.length ? "info" : "success");
        }
        if (job.status === "failed") toast(kind === "check" ? "Check failed" : "Upload failed", job.error || "", "error");
        state.watching = false;
        await openProject(state.project.project_id);
        if (job.status === "done" && kind === "check") selectTab("compliance");
        await loadProjects();
        return job;
      }
      await new Promise((r) => setTimeout(r, 900));
    }
  } finally { state.watching = false; btn.disabled = false; }
}

async function runCheck() {
  const sample = parseInt($("sample").value || "0", 10);
  const budget = parseFloat($("budget").value || "1");
  try {
    const r = await postJSON(`/projects/${state.project.project_id}/check`, { sample, budget_usd: budget });
    toast("Check started", `Budget cap $${r.budget_usd.toFixed(2)}${sample ? ` · sample of ${sample}` : ""}`);
    await watchJob(r.job_id, "check");
  } catch (e) { toast("Could not start the check", e.message, "error"); }
}

// ---------------------------------------------------------------- KPIs + matrix

function renderNoResults() {
  $("results").hidden = true;
  $("noResults").hidden = false;
  $("kpis").hidden = true;
  $("coverage").hidden = true;
  $("tabCount").hidden = true;
}

async function loadMatrix() {
  const m = await getJSON(`/projects/${state.project.project_id}/matrix?format=json`);
  state.matrix = m;
  $("results").hidden = false;
  $("noResults").hidden = true;
  const counts = { MISSING: 0, PARTIAL: 0, UNCLEAR: 0, MET: 0 };
  const mand = { MISSING: 0, PARTIAL: 0, UNCLEAR: 0, MET: 0 };
  m.matrix.forEach((r) => { counts[r.status]++; if (r.mandatory) mand[r.status]++; });
  renderKpis(counts, mand, m.matrix.length);
  const s = m.stats;
  $("checkMeta").textContent = `Check ${m.check_id} · ${m.matrix.length} judged${s.sampled ? " (sample)" : ""} · ` +
    `${s.llm_calls || 0} model calls · ${fmtUsd(s.cost_usd)}`;
  const warns = m.warnings || [];
  $("warnings").hidden = !warns.length;
  $("warningList").replaceChildren(...warns.map((w) => el("li", { dir: "auto" }, w)));
  const cats = [...new Set(m.matrix.map((r) => r.category))].sort();
  const cur = $("fCat").value;
  $("fCat").replaceChildren(el("option", { value: "" }, "All categories"), ...cats.map((c) => el("option", { value: c }, c.replace("_", " "))));
  $("fCat").value = cats.includes(cur) ? cur : "";
  $("tabCount").hidden = !counts.MISSING;
  $("tabCount").textContent = counts.MISSING;
  renderMatrix();
}

function renderKpis(counts, mand, total) {
  const kpis = $("kpis");
  kpis.hidden = false;
  const label = { MISSING: "Missing", PARTIAL: "Partial", UNCLEAR: "Unclear", MET: "Met" };
  kpis.replaceChildren(...STATUSES.map((s) => el("button", { class: `card kpi kpi-${s}`, type: "button",
    title: `Show ${label[s].toLowerCase()} requirements`,
    onclick: () => { $("fStatus").value = s; selectTab("compliance"); renderMatrix(); } },
  el("div", { class: "kpi-top" }, label[s], icon(STATUS_ICON[s])),
  el("div", { class: "kpi-value" }, counts[s]),
  el("div", { class: "kpi-sub" }, `${mand[s]} mandatory · ${total ? Math.round(100 * counts[s] / total) : 0}%`))));
  $("coverage").hidden = false;
  const bar = $("coverageBar");
  bar.replaceChildren(...["MET", "PARTIAL", "UNCLEAR", "MISSING"].filter((s) => counts[s]).map((s) => el("span", {
    class: `seg-${s}`, style: `width:${(100 * counts[s] / total).toFixed(2)}%`, title: `${counts[s]} ${s}` })));
  bar.setAttribute("aria-label", STATUSES.map((s) => `${counts[s]} ${s}`).join(", "));
  $("coverageMeta").textContent = `${counts.MET} of ${total} met · ${counts.MISSING + counts.PARTIAL} need work`;
}

function rowsFiltered() {
  const m = state.matrix;
  const st = $("fStatus").value, cat = $("fCat").value, mand = $("fMand").checked, q = $("fText").value.toLowerCase();
  let rows = m.matrix.filter((r) => (!st || r.status === st) && (!cat || r.category === cat) && (!mand || r.mandatory) &&
    (!q || `${r.req_id} ${r.requirement} ${r.tender_quote} ${r.bid_quote} ${r.reason}`.toLowerCase().includes(q)));
  const { key, dir } = state.sort;
  if (key === "req_id") rows = rows.slice().sort((a, b) => dir * a.req_id.localeCompare(b.req_id));
  if (key === "status") rows = rows.slice().sort((a, b) => dir * (STATUSES.indexOf(a.status) - STATUSES.indexOf(b.status)));
  return rows;
}

function renderMatrix() {
  if (!state.matrix) return;
  const rows = rowsFiltered();
  document.querySelectorAll(".th-sort").forEach((b) => b.parentElement.setAttribute("aria-sort",
    state.sort.key === b.dataset.sort ? (state.sort.dir > 0 ? "ascending" : "descending") : "none"));
  $("rowCount").textContent = `${rows.length} of ${state.matrix.matrix.length}`;
  const body = $("matrixBody");
  if (!rows.length) { body.replaceChildren(el("tr", { class: "table-empty" }, el("td", { colspan: 6 }, "No requirements match these filters."))); return; }
  body.replaceChildren(...rows.map((r) => el("tr", {},
    el("td", { class: "col-req" }, r.req_id, r.mandatory ? null : el("span", { class: "opt" }, "optional")),
    el("td", {}, statusBadge(r.status),
      r.reviewed ? el("div", { class: "flag" }, "re-checked") : null,
      r.downgraded ? el("div", { class: "flag" }, "quote failed check") : null),
    el("td", { class: "col-requirement" }, el("div", { class: "req-text", dir: "auto" }, r.requirement),
      el("div", { class: "req-meta" }, el("span", { class: "badge badge-outline" }, r.category.replace("_", " ")),
        r.tender_section ? el("span", { class: "badge badge-outline", dir: "auto", title: r.tender_section }, r.tender_section.slice(0, 40)) : null)),
    el("td", {}, el("button", { class: "cite", type: "button", title: "Open the tender page with the quote highlighted",
      onclick: () => openViewer(r.tender_doc_id, r.tender_page, r.tender_quote, `Tender · ${r.req_id}`) }, icon("file"), `p. ${r.tender_page}`)),
    el("td", { class: "col-evidence" }, r.bid_quote
      ? [el("div", { class: "quote", dir: "auto" }, `“${r.bid_quote}”`),
        el("div", { class: "cite-row" }, el("button", { class: "cite", type: "button", title: "Open the bid page with the quote highlighted",
          onclick: () => openViewer(r.bid_doc_id, r.bid_page, r.bid_quote, `Bid · ${r.req_id}`) }, icon("file"), `${r.bid_file} · p. ${r.bid_page}`))]
      : el("span", { class: "muted" }, "No evidence in the bid")),
    el("td", { class: "col-reason", dir: "auto" }, r.reason))));
}

async function exportMatrix(fmt) {
  const res = await api(`/projects/${state.project.project_id}/matrix?format=${fmt}`);
  const blob = await res.blob();
  const a = el("a", { href: URL.createObjectURL(blob), download: `compliance_${state.project.project_id}.${fmt}` });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  toast(`Exported ${fmt.toUpperCase()}`, "", "success");
}

// ---------------------------------------------------------------- citation sheet

function openViewer(docId, page, quote, title) {
  if (!docId) return;
  const doc = state.project.documents.find((d) => d.doc_id === docId);
  state.viewer = { docId, page, quote, title, pages: doc ? doc.pages : page, isPdf: !!doc && doc.filename.toLowerCase().endsWith(".pdf"),
    filename: doc ? doc.filename : "" };
  state.lastFocus = document.activeElement;
  $("sheetScrim").hidden = false;
  $("viewer").hidden = false;
  $("closeViewer").focus();
  renderViewer();
}
function closeViewer() {
  $("viewer").hidden = true;
  $("sheetScrim").hidden = true;
  if (state.lastFocus && state.lastFocus.focus) state.lastFocus.focus();
}

async function renderViewer() {
  const v = state.viewer;
  $("viewerTitle").textContent = v.title;
  $("viewerSub").textContent = v.filename;
  $("viewerPage").textContent = `Page ${v.page} of ${v.pages}`;
  $("prevPage").disabled = v.page <= 1;
  $("nextPage").disabled = v.page >= v.pages;
  const body = $("viewerBody");
  body.replaceChildren(el("div", { class: "skeleton", style: "height:520px" }));
  const base = `/projects/${state.project.project_id}/documents/${v.docId}/pages/${v.page}`;
  const q = encodeURIComponent(v.quote || "");
  try {
    if (v.isPdf) {
      const res = await api(`${base}/image?q=${q}`);
      const hits = res.headers.get("X-Highlights");
      const img = el("img", { alt: `Page ${v.page} of ${v.filename}`, src: URL.createObjectURL(await res.blob()) });
      const note = !v.quote ? null : hits === "0"
        ? el("p", { class: "sheet-note" }, icon("info", "icon icon-xs"), "The quote was verified on this page but could not be highlighted exactly.")
        : el("p", { class: "sheet-note" }, icon("eye", "icon icon-xs"), "The cited text is highlighted.");
      body.replaceChildren(...[note, img].filter(Boolean));
    } else {
      const t = await getJSON(`${base}/text?q=${q}`);
      const pre = el("pre", { dir: "auto" });
      if (t.highlight) {
        const [a, b] = t.highlight;
        pre.append(t.text.slice(0, a), el("mark", {}, t.text.slice(a, b)), t.text.slice(b));
      } else pre.textContent = t.text;
      body.replaceChildren(el("p", { class: "sheet-note" }, icon("info", "icon icon-xs"),
        "Word documents have no fixed pages: page numbers follow the breaks Word saved."), pre);
      const mark = pre.querySelector("mark");
      if (mark) mark.scrollIntoView({ block: "center" });
    }
  } catch (e) { body.replaceChildren(el("p", { class: "muted" }, `Could not load the page: ${e.message}`)); }
}

function pageStep(delta) {
  const v = state.viewer;
  if (!v) return;
  const next = v.page + delta;
  if (next < 1 || next > v.pages) return;
  v.page = next;
  renderViewer();
}

// ---------------------------------------------------------------- ask (chat over SSE)

function chatFor() { const id = state.project.project_id; return (state.chats[id] = state.chats[id] || []); }

function renderChat() {
  const log = $("chatLog");
  const items = state.project ? chatFor() : [];
  [...log.querySelectorAll(".msg")].forEach((n) => n.remove());
  log.querySelector(".chat-empty").hidden = items.length > 0;
  items.forEach((m) => log.appendChild(m.node));
}

function addMsg(node) {
  chatFor().push({ node });
  $("chatLog").querySelector(".chat-empty").hidden = true;
  $("chatLog").appendChild(node);
  node.scrollIntoView({ block: "end", behavior: "smooth" });
}

function botNode() {
  const bubble = el("div", { class: "bubble" }, el("span", { class: "typing", "aria-label": "Answering" }, el("span"), el("span"), el("span")));
  const label = el("div", { class: "msg-label" }, icon("shield", "icon icon-xs"), "Searching the tender and bid…");
  return { node: el("div", { class: "msg msg-bot" }, label, bubble), label, bubble };
}

async function ask(ev) {
  ev.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  const scope = $("scope").value;
  $("question").value = "";
  addMsg(el("div", { class: "msg msg-user" }, el("div", { class: "bubble", dir: "auto" }, question)));
  const bot = botNode();
  addMsg(bot.node);
  $("askBtn").disabled = true;
  try {
    const res = await api(`/projects/${state.project.project_id}/ask`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, scope }) });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, i); buf = buf.slice(i + 2);
        const evName = (raw.match(/^event: (.*)$/m) || [])[1];
        const data = JSON.parse((raw.match(/^data: (.*)$/m) || [])[1] || "{}");
        if (evName === "status") bot.label.replaceChildren(icon("shield", "icon icon-xs"), data.stage.charAt(0).toUpperCase() + data.stage.slice(1) + "…");
        if (evName === "error") { bot.label.replaceChildren(icon("x", "icon icon-xs"), "Error"); bot.bubble.textContent = data.message; }
        if (evName === "answer") renderAnswer(bot, data);
      }
    }
  } catch (e) { bot.label.replaceChildren(icon("x", "icon icon-xs"), "Error"); bot.bubble.textContent = e.message; }
  finally { $("askBtn").disabled = false; $("question").focus(); }
}

function renderAnswer(bot, a) {
  const docs = state.project.documents;
  bot.label.replaceChildren(icon("shield", "icon icon-xs"),
    a.citations.length ? `Answer · ${plural(a.citations.length, "verified citation")}` : "Answer");
  bot.bubble.replaceChildren(el("div", { dir: "auto" }, a.answer));
  if (a.citations.length) {
    bot.bubble.appendChild(el("div", { class: "msg-cites" }, ...a.citations.map((c) => {
      const d = docs.find((x) => x.filename === c.filename);
      return el("div", { class: "msg-cite" },
        el("button", { class: "cite", type: "button", onclick: () => d && openViewer(d.doc_id, c.page, c.quote, `${c.role === "tender" ? "Tender" : "Bid"} · ${c.excerpt_id}`) },
          icon("file"), `${c.role} · p. ${c.page}`),
        el("span", { dir: "auto" }, `“${c.quote}”`));
    })));
  }
  for (const issue of (a.issues || []).concat(a.refused ? [a.refused] : [])) bot.bubble.appendChild(el("div", { class: "msg-issue" }, issue));
}

// ---------------------------------------------------------------- sidebar (mobile)

function openSidebar() { document.querySelector(".app").classList.add("sidebar-open"); $("sidebarScrim").hidden = false; }
function closeSidebar() { document.querySelector(".app").classList.remove("sidebar-open"); $("sidebarScrim").hidden = true; }

// ---------------------------------------------------------------- wiring

function wire() {
  let pref = "system";
  try { pref = localStorage.getItem("tenderTheme") || "system"; } catch { /* default */ }
  applyTheme(pref);
  document.querySelectorAll("[data-theme-set]").forEach((b) => { b.onclick = () => applyTheme(b.dataset.themeSet); });

  $("userBtn").onclick = askKey;
  $("keyForm").onsubmit = (e) => { e.preventDefault(); saveKey($("keyInput").value); closeDialogs(); boot(); };
  const openNew = () => { $("newName").value = ""; openDialog("newDialog"); };
  $("newProjectBtn").onclick = openNew;
  $("emptyNewBtn").onclick = openNew;
  $("newProject").onsubmit = async (e) => {
    e.preventDefault();
    try {
      const p = await postJSON("/projects", { name: $("newName").value });
      closeDialogs();
      toast("Project created", p.name, "success");
      await loadProjects(p.project_id);
      selectTab("overview");
    } catch (err) { toast("Could not create the project", err.message, "error"); }
  };
  $("deleteBtn").onclick = () => openDialog("deleteDialog");
  $("confirmDelete").onclick = async () => {
    const p = state.project;
    try {
      await api(`/projects/${p.project_id}`, { method: "DELETE" });
      closeDialogs();
      state.project = null; state.matrix = null;
      $("projectView").hidden = true; $("empty").hidden = false; $("crumbProject").textContent = "Overview";
      closeViewer();
      toast("Project deleted", p.name, "success");
      loadProjects();
    } catch (err) { toast("Could not delete the project", err.message, "error"); }
  };
  document.querySelectorAll("[data-close]").forEach((b) => { b.onclick = closeDialogs; });
  $("dialogScrim").onclick = closeDialogs;

  document.querySelectorAll(".tab").forEach((t) => { t.onclick = () => selectTab(t.dataset.tab); });
  document.querySelector(".tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const tabs = [...document.querySelectorAll(".tab")];
    const i = tabs.findIndex((t) => t.dataset.tab === state.tab);
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    selectTab(next.dataset.tab); next.focus();
  });

  document.querySelectorAll(".dropzone").forEach((d) => {
    const input = d.querySelector("input");
    input.onchange = () => { upload(d.dataset.role, [...input.files]); input.value = ""; };
    d.ondragover = (e) => { e.preventDefault(); d.classList.add("over"); };
    d.ondragleave = () => d.classList.remove("over");
    d.ondrop = (e) => { e.preventDefault(); d.classList.remove("over"); upload(d.dataset.role, [...e.dataTransfer.files]); };
  });
  $("checkBtn").onclick = runCheck;

  ["fStatus", "fCat", "fMand", "fText"].forEach((id) => $(id).addEventListener("input", renderMatrix));
  document.querySelectorAll(".th-sort").forEach((b) => {
    b.onclick = () => {
      state.sort = { key: b.dataset.sort, dir: state.sort.key === b.dataset.sort ? -state.sort.dir : 1 };
      renderMatrix();
    };
  });
  document.querySelectorAll("[data-export]").forEach((b) => { b.onclick = () => exportMatrix(b.dataset.export).catch((e) => toast("Export failed", e.message, "error")); });

  $("closeViewer").onclick = closeViewer;
  $("sheetScrim").onclick = closeViewer;
  $("prevPage").onclick = () => pageStep(-1);
  $("nextPage").onclick = () => pageStep(1);

  $("askForm").onsubmit = ask;
  document.querySelectorAll("[data-q]").forEach((c) => { c.onclick = () => { $("question").value = c.dataset.q; $("askForm").requestSubmit(); }; });

  $("projectSearch").addEventListener("input", renderProjects);
  $("menuBtn").onclick = openSidebar;
  $("sidebarScrim").onclick = closeSidebar;

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (anyDialogOpen()) closeDialogs();
    else if (!$("viewer").hidden) closeViewer();
    else closeSidebar();
  });
}

loadKey();
wire();
boot();
