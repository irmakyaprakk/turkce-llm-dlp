"use strict";

// Yönetici paneli: /v1/audit/summary verisini çizer. Dış kütüphane yok (CSP: script-src 'self').
// Tüm metinler textContent ile yazılır; sunucudan gelen hiçbir değer HTML olarak yorumlanmaz.

const TYPE_LABELS = {
  PERSON: "Kişi", ORG: "Kurum", LOCATION: "Yer", ADDRESS: "Adres", TCKN: "T.C. kimlik no",
  IBAN: "IBAN", CARD: "Kredi kartı", EMAIL: "E-posta", PHONE: "Telefon", SECRET: "Gizli bilgi"
};
const OUTCOME_LABELS = {
  masked: "Maskelendi",
  clean: "Temiz",
  blocked: "Engellendi",
  cancelled: "Vazgeçildi",
  leak_detected: "Kontrolsüz gönderim"
};
const ACTION_LABELS = {mask: "Maskele", block: "Engelle", allow: "İzin ver"};
const SVG_NS = "http://www.w3.org/2000/svg";
const fmt = new Intl.NumberFormat("tr-TR");
const dayFmt = new Intl.DateTimeFormat("tr-TR", {day: "numeric", month: "short"});
const timeFmt = new Intl.DateTimeFormat("tr-TR", {day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"});

let days = 30;
const $ = (id) => document.getElementById(id);
const tooltip = $("tooltip");

function readToken() {
  try { return sessionStorage.getItem("turkce-dlp-admin") || ""; } catch (_e) { return ""; }
}
function writeToken(value) {
  try { sessionStorage.setItem("turkce-dlp-admin", value); } catch (_e) { /* yalnız bu sekme için */ }
}

function el(tag, attrs = {}, text) {
  const node = tag.startsWith("svg:") ? document.createElementNS(SVG_NS, tag.slice(4)) : document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function sum(counts) {
  return Object.values(counts || {}).reduce((a, b) => a + b, 0);
}

function summarizeCounts(counts) {
  const parts = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${TYPE_LABELS[k] || k} ${v}`);
  return parts.length ? parts.join(", ") : "—";
}

function empty(container, message) {
  container.replaceChildren(el("div", {class: "empty"}, message));
}

// Tablo: rows = [[hücre, ...], ...]; sayısal sütunlar sağa yaslanır.
function table(container, headers, rows, numeric = []) {
  if (!rows.length) return empty(container, "Henüz kayıt yok");
  const t = el("table");
  const head = el("tr");
  headers.forEach((h, i) => head.append(el("th", numeric.includes(i) ? {class: "num"} : {}, h)));
  t.append(el("thead"), el("tbody"));
  t.tHead.append(head);
  for (const row of rows) {
    const tr = el("tr");
    row.forEach((cell, i) => {
      const td = el("td", numeric.includes(i) ? {class: "num"} : {});
      if (cell instanceof Node) td.append(cell); else td.textContent = cell;
      tr.append(td);
    });
    t.tBodies[0].append(tr);
  }
  const wrap = el("div", {class: "scroll"});
  wrap.append(t);
  container.replaceChildren(wrap);
}

function tag(text, critical = false) {
  return el("span", {class: critical ? "tag block" : "tag"}, text);
}

// --- ipucu kutusu ---
function showTip(event, title, lines) {
  tooltip.replaceChildren(el("strong", {}, title));
  for (const line of lines) tooltip.append(el("div", {}, line));
  tooltip.style.display = "block";
  const pad = 14;
  const box = tooltip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + box.width > window.innerWidth - 8) x = event.clientX - box.width - pad;
  if (y + box.height > window.innerHeight - 8) y = event.clientY - box.height - pad;
  tooltip.style.left = `${Math.max(8, x)}px`;
  tooltip.style.top = `${Math.max(8, y)}px`;
}
function hideTip() {
  tooltip.style.display = "none";
}

// Tam sayılı, okunaklı eksen adımı (1, 2, 5 × 10^n) ve ona yuvarlanmış üst sınır
function niceScale(value, target = 4) {
  const raw = Math.max(1, value / target);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((candidate) => candidate >= raw);
  return {step, max: step * Math.max(1, Math.ceil(value / step))};
}

// Üstü 4px yuvarlatılmış, tabana oturan çubuk
function barPath(x, y, w, h) {
  const r = Math.min(4, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

// --- günlük sütun grafiği (tek seri) ---
function dailyChart(container, daily) {
  if (!daily.length || daily.every((d) => d.messages === 0)) return empty(container, "Bu dönemde kayıt yok");
  const W = 640, H = 230, left = 34, right = 8, top = 12, bottom = 26;
  const plotW = W - left - right, plotH = H - top - bottom;
  const {step, max} = niceScale(Math.max(...daily.map((d) => d.masked), 1));
  const slot = plotW / daily.length;
  const barW = Math.max(2, Math.min(24, slot - 2));
  const svg = el("svg:svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Günlük maskelenen kişisel veri sayısı"});

  for (let value = 0; value <= max; value += step) {
    const y = top + plotH - (value / max) * plotH;
    svg.append(el("svg:line", {x1: left, x2: W - right, y1: y, y2: y, class: value === 0 ? "baseline" : "gridline"}));
    svg.append(el("svg:text", {x: left - 6, y: y + 4, "text-anchor": "end"}, fmt.format(Math.round(value))));
  }

  const labelEvery = Math.ceil(daily.length / 6);
  daily.forEach((d, i) => {
    const x = left + i * slot + (slot - barW) / 2;
    const h = (d.masked / max) * plotH;
    const date = new Date(`${d.day}T12:00:00`);
    const bar = el("svg:path", {d: barPath(x, top + plotH - h, barW, Math.max(h, 0)), class: "bar"});
    const hit = el("svg:rect", {x: left + i * slot, y: top, width: slot, height: plotH, class: "hit"});
    hit.addEventListener("mousemove", (event) => {
      bar.classList.add("active");
      showTip(event, dayFmt.format(date), [
        `Maskelenen: ${fmt.format(d.masked)}`,
        `Kontrol edilen mesaj: ${fmt.format(d.messages)}`,
        `Engellenen: ${fmt.format(d.blocked)}`
      ]);
    });
    hit.addEventListener("mouseleave", () => { bar.classList.remove("active"); hideTip(); });
    svg.append(bar, hit);
    if ((daily.length - 1 - i) % labelEvery === 0) {
      svg.append(el("svg:text", {x: x + barW / 2, y: H - 8, "text-anchor": "middle"}, dayFmt.format(date)));
    }
  });
  container.replaceChildren(svg);
}

// --- türe göre yatay çubuklar (doğrudan etiketli) ---
function typeChart(container, masked) {
  const entries = Object.entries(masked || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return empty(container, "Bu dönemde maskelenen veri yok");
  const rowH = 30, labelW = 110, valueW = 44, W = 380;
  const H = entries.length * rowH;
  const max = entries[0][1];
  const svg = el("svg:svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Türe göre maskelenen veri"});
  entries.forEach(([kind, count], i) => {
    const y = i * rowH;
    const w = Math.max(2, (count / max) * (W - labelW - valueW));
    const label = TYPE_LABELS[kind] || kind;
    svg.append(el("svg:text", {x: 0, y: y + rowH / 2 + 4, class: "cat-label"}, label));
    const bar = el("svg:path", {d: barPathH(labelW, y + 7, w, rowH - 14), class: "bar"});
    const hit = el("svg:rect", {x: 0, y, width: W, height: rowH, class: "hit"});
    hit.addEventListener("mousemove", (event) => {
      bar.classList.add("active");
      showTip(event, label, [`Maskelenen: ${fmt.format(count)}`]);
    });
    hit.addEventListener("mouseleave", () => { bar.classList.remove("active"); hideTip(); });
    svg.append(bar, el("svg:text", {x: labelW + w + 6, y: y + rowH / 2 + 4, class: "value-label"}, fmt.format(count)), hit);
  });
  container.replaceChildren(svg);
}

// Sağ ucu 4px yuvarlatılmış yatay çubuk (taban solda)
function barPathH(x, y, w, h) {
  const r = Math.min(4, h / 2, w);
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}

function render(data) {
  const maskedTotal = sum(data.masked);
  $("k-messages").textContent = fmt.format(data.messages);
  const siteCount = Object.keys(data.sites || {}).length;
  $("k-messages-note").textContent = `Son ${data.days} gün · ${siteCount} site`;
  $("k-masked").textContent = fmt.format(maskedTotal);
  $("k-blocked").textContent = fmt.format(data.outcomes?.blocked || 0);
  $("k-fp").textContent = data.false_positive_rate === null || data.false_positive_rate === undefined
    ? "—" : `%${(data.false_positive_rate * 100).toLocaleString("tr-TR", {maximumFractionDigits: 1})}`;
  $("model-chip").textContent = data.model ? "Model: BERTurk + kurallar" : "Model: yalnız kurallar";

  dailyChart($("daily-chart"), data.daily || []);
  typeChart($("type-chart"), data.masked);

  const fpRows = Object.keys(data.detected || {})
    .sort((a, b) => (data.unchecked?.[b] || 0) - (data.unchecked?.[a] || 0) || data.detected[b] - data.detected[a])
    .map((kind) => {
      const found = data.detected[kind];
      const off = data.unchecked?.[kind] || 0;
      return [TYPE_LABELS[kind] || kind, fmt.format(found), fmt.format(off), found ? `%${Math.round((off / found) * 100)}` : "—"];
    });
  table($("fp-table"), ["Tür", "Bulunan", "Kaldırılan", "Oran"], fpRows, [1, 2, 3]);

  const total = data.messages || 0;
  const outcomeRows = ["masked", "clean", "cancelled", "blocked", "leak_detected"]
    .filter((key) => data.outcomes?.[key])
    .map((key) => [tag(OUTCOME_LABELS[key], key === "blocked" || key === "leak_detected"), fmt.format(data.outcomes[key]),
      `%${Math.round((data.outcomes[key] / total) * 100)}`]);
  table($("outcome-table"), ["Karar", "Mesaj", "Pay"], outcomeRows, [1, 2]);

  const policyRows = Object.entries(data.policy?.rules || {})
    .sort((a, b) => (TYPE_LABELS[a[0]] || a[0]).localeCompare(TYPE_LABELS[b[0]] || b[0], "tr"))
    .map(([kind, action]) => [TYPE_LABELS[kind] || kind, tag(ACTION_LABELS[action] || action, action === "block")]);
  table($("policy-table"), ["Tür", "Eylem"], policyRows);

  const recentRows = (data.recent || []).map((row) => [
    timeFmt.format(new Date(row.ts * 1000)),
    row.site,
    tag(OUTCOME_LABELS[row.outcome] || row.outcome, row.outcome === "blocked" || row.outcome === "leak_detected"),
    summarizeCounts(row.masked),
    sum(row.unchecked) ? fmt.format(sum(row.unchecked)) : "—"
  ]);
  table($("recent-table"), ["Zaman", "Site", "Karar", "Maskelenen", "Kaldırılan"], recentRows, [4]);
}

async function load() {
  const token = readToken();
  let response;
  try {
    response = await fetch(`/v1/audit/summary?days=${days}`, {
      headers: token ? {"X-Admin-Token": token} : {},
      cache: "no-store"
    });
  } catch (_error) {
    return empty($("daily-chart"), "Servise ulaşılamadı");
  }
  if (response.status === 401) {
    $("auth").style.display = "block";
    $("token").focus();
    return;
  }
  $("auth").style.display = "none";
  if (response.status === 404) {
    for (const id of ["daily-chart", "type-chart", "fp-table", "outcome-table", "recent-table"]) {
      empty($(id), "Denetim günlüğü kapalı (TURKCE_DLP_AUDIT=0)");
    }
    return;
  }
  if (!response.ok) return empty($("daily-chart"), "Özet alınamadı");
  render(await response.json());
}

$("auth").addEventListener("submit", (event) => {
  event.preventDefault();
  writeToken($("token").value);
  $("token").value = "";
  load();
});

for (const button of document.querySelectorAll("[data-days]")) {
  button.addEventListener("click", () => {
    days = Number(button.dataset.days);
    for (const other of document.querySelectorAll("[data-days]")) other.setAttribute("aria-pressed", String(other === button));
    load();
  });
}

load();
setInterval(() => { if (document.visibilityState === "visible") load(); }, 30_000);
