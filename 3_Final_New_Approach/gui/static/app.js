// app.js - DINO-CPS plant disease demo, fully in the browser (Hugging Face static Space).
// Flow: photo -> plain resize 224x224 -> ImageNet normalise -> 4 ONNX models (ONNX Runtime Web, WebAssembly):
//   DINO-CPS crop + DINO-CPS disease (int8 DINOv2 ViT-L) and team ConvNeXt-T crop + disease.
// Pair score, identical to 3_Final_New_Approach/training/evaluate.py (+prior+CNN; b, q, w tuned on validation by build_assets.py):
//   s[c,d] = log p_dino(c) + log p_dino(d) + w * (log p_cnn(c) + log p_cnn(d)) - b * seen + q * host_prior
// Upload, examples, top-5 bars and confusion matrix are reused from the earlier GUI.

const $ = (id) => document.getElementById(id);
let meta, ev, P;                      // data/meta.json, data/eval.json, data/params.json
let last = null;                      // { canvas, att } for the attention overlay
let pending = null;                   // photo chosen before the models finished loading
const S = {};                         // ONNX sessions
const MODELS = [["dino_cps_crop", "models/dino_cps_crop.int8.onnx"], ["dino_cps_disease", "models/dino_cps_disease.int8.onnx"],
  ["cnn_crop", "models/crop_ConvNeXt-Tiny.onnx"], ["cnn_disease", "models/disease_ConvNeXt-Tiny.onnx"]];
const MODEL_CACHE = "dino-cps-models-1";      // bump after every re-export of the .onnx files
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const pct = (v) => (v == null ? "–" : `${v.toFixed(1)}%`);
const nice = (s) => s.replaceAll("_", " ");
const argsort = (a) => [...a.keys()].sort((i, j) => a[j] - a[i]);

function status(text, frac = null) {
  $("status").classList.toggle("done", text == null);
  if (text != null) $("statusText").textContent = text;
  if (frac != null) $("loadBar").style.width = `${Math.round(frac * 100)}%`;
}

// Download with progress. Models are kept in the Cache API, so a second visit does not download ~720 MB again.
async function getBytes(url, onProgress) {
  let cache = null;
  try {
    for (const k of await caches.keys()) if (k !== MODEL_CACHE) await caches.delete(k);   // drop old exports
    cache = await caches.open(MODEL_CACHE); const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); }
  catch { cache = null; }                       // private mode / no Cache API: just download
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get("Content-Length") || 0, reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; if (total) onProgress(got / total); }
  const out = new Uint8Array(got); let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  try { await cache?.put(url, new Response(out)); } catch { /* quota full: fine, it downloads next time */ }
  return out;
}

async function loadModels() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  for (const [i, [k, url]] of MODELS.entries()) {
    const bytes = await getBytes(url, (x) => status(`Loading model ${i + 1} of 4 (first visit only, about 720 MB in total)…`, (i + x) / 4));
    status(`Starting model ${i + 1} of 4…`, (i + 1) / 4);
    S[k] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  }
  status(null);
  if (pending) { const p = pending; pending = null; run(...p); }
}

async function init() {
  [meta, ev, P] = await Promise.all(["data/meta.json", "data/eval.json", "data/params.json"].map((u) => fetch(u).then((r) => r.json())));
  meta.unseen = new Set(meta.unseen_comp_ids);
  meta.trained = new Set(meta.train_comp_ids);
  renderStats(); renderExamples(); renderEval(); renderCM();
  const t = ev.tuned;
  $("tuned").textContent = `Deployed values, tuned on validation only: w = ${t.w}, b = ${t.b.toFixed(2)}, host-prior weight q = ${t.q} (val H ${t.val_H.toFixed(1)}).`;
  await loadModels();
}

function renderStats() {
  const v = ev.rows.find((r) => r.ours);
  $("stats").innerHTML = [
    ["Pair Top-1 seen", pct(v.top1.seen)], ["Pair Top-1 unseen", pct(v.top1.unseen)], ["H Top-1", pct(v.top1.H)],
    ["H Top-5", pct(v.top5.H)], ["Disease alone, unseen", pct(v.disease_unseen)],
  ].map(([k, x]) => `<div class="stat"><b>${x}</b><span>${k}</span></div>`).join("");
}

// ---------------------------------------------------------------- input
function renderExamples() {
  for (const ex of meta.examples) {
    const b = document.createElement("button");
    b.title = `${nice(meta.comp[ex.comp])} (${ex.split})`;
    b.innerHTML = `<img src="${ex.file}" alt="Test image: ${nice(meta.comp[ex.comp])}" loading="lazy">`;
    b.onclick = () => fetch(ex.file).then((r) => r.blob()).then((blob) => run(blob, ex.comp));
    $(ex.split === "test_seen" ? "exSeen" : "exUnseen").appendChild(b);
  }
}
const loadFile = (f) => f && f.type.startsWith("image/") && run(f, null);
$("file").addEventListener("change", (e) => loadFile(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("file").click(); });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) loadFile(item.getAsFile());
});

// ---------------------------------------------------------------- prediction
let reqId = 0;
async function run(blob, truth) {
  const id = ++reqId;
  $("err").hidden = true; $("empty").hidden = true; $("busy").hidden = false;
  const canvas = await toCanvas(blob).catch(() => null);
  if (!canvas) return fail("This file could not be opened as an image.");
  last = { canvas, att: null }; $("previewBox").hidden = false; drawView();
  if (Object.keys(S).length < MODELS.length) { pending = [blob, truth]; $("busy").textContent = "Waiting for the models to load…"; return; }
  $("busy").innerHTML = '<span class="spin" aria-hidden="true"></span> Running DINO-CPS in your browser (about 5–10 s)…';
  await new Promise((r) => setTimeout(r, 30));      // let the spinner paint before the long WebAssembly call
  let out;
  try { out = await predict(canvas); } catch (e) { return id === reqId && fail(`Prediction failed: ${e.message}`); }
  if (id !== reqId) return;                       // a newer photo was chosen meanwhile
  $("busy").hidden = true;
  last.att = out.att; drawView();
  showResult(out, truth);
}
function fail(msg) { $("busy").hidden = true; $("err").textContent = msg; $("err").hidden = false; }

// Plain resize to 224x224 (no crop), the same as the dataset and the evaluation.
async function toCanvas(blob) {
  const bmp = await createImageBitmap(blob);
  const c = document.createElement("canvas"); c.width = c.height = 224;
  const ctx = c.getContext("2d", { willReadFrequently: true }); ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, 224, 224);
  return c;
}

const softmax = (a) => { const m = Math.max(...a), e = a.map((v) => Math.exp(v - m)), z = e.reduce((x, y) => x + y, 0); return e.map((v) => v / z); };

async function predict(canvas) {
  const px = canvas.getContext("2d").getImageData(0, 0, 224, 224).data, n = 224 * 224, data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) data[ch * n + i] = (px[i * 4 + ch] / 255 - MEAN[ch]) / STD[ch];
  const input = new ort.Tensor("float32", data, [1, 3, 224, 224]);
  const t0 = performance.now(), o = {};
  for (const k of Object.keys(S)) Object.assign(o, await S[k].run({ input }));
  const lc = [...o.crop_logp.data], ld = [...o.disease_logp.data];
  const cc = Array.from(o.crop_probs.data, (p) => Math.log(p + 1e-9)), cd = Array.from(o.disease_probs.data, (p) => Math.log(p + 1e-9));
  const score = P.cd.map(([c, d], i) => lc[c] + ld[d] + P.w * (cc[c] + cd[d]) - P.b * P.seen[i] + P.q * P.sup[i]);
  return { pair: softmax(score), crop: lc.map(Math.exp), disease: ld.map(Math.exp), att: [...o.att.data], ms: Math.round(performance.now() - t0) };
}

function bars(el, probs, names, trueId, n = 5) {
  el.innerHTML = argsort(probs).slice(0, n).map((i) => `
    <li class="${i === trueId ? "truth" : ""}"><div>
      <div class="row"><span>${nice(names[i])}${names === meta.comp && !meta.trained.has(i) ? ' <span class="tag unseen">not in training</span>' : ""}</span><span>${(probs[i] * 100).toFixed(1)}%</span></div>
      <div class="track"><div class="fill" style="width:${(probs[i] * 100).toFixed(1)}%"></div></div>
    </div></li>`).join("");
}

function showResult(out, truth) {
  $("result").hidden = false;
  const top = argsort(out.pair)[0];
  $("top1Name").textContent = nice(meta.comp[top]);
  $("top1Conf").textContent = `${(out.pair[top] * 100).toFixed(1)}% (pair score, normalised over 333 pairs)`;
  $("lowConf").hidden = out.pair[top] >= 0.4;
  const [tc, td] = truth == null ? [null, null] : meta.comp_to_crop_disease[truth];
  if (truth == null) $("truth").innerHTML = "";
  else {
    const tag = meta.unseen.has(truth) ? '<span class="tag unseen">unseen pair</span>' : '<span class="tag seen">seen pair</span>';
    $("truth").innerHTML = (truth === top ? '<span class="ok">✓ Correct</span> ' : `<span class="bad">✗ True: ${nice(meta.comp[truth])}</span> `) + tag;
  }
  renderGallery(top);
  bars($("topPair"), out.pair, meta.comp, truth);
  bars($("topCrop"), out.crop, meta.crop, tc);
  bars($("topDisease"), out.disease, meta.disease, td);
  $("timing").textContent = `Inference ${(out.ms / 1000).toFixed(1)} s in your browser (2 × int8 DINOv2 ViT-L + 2 × ConvNeXt-T, WebAssembly). Nothing was uploaded.`;
}

// Brief: "display the relevant training images it matches".
// Seen pair: its own training images. Pair not in training: the same disease on other crops + other images of the same crop.
function renderGallery(top) {
  const [c, d] = meta.comp_to_crop_disease[top];
  let items, note;
  if (meta.gallery[top]) {
    items = meta.gallery[top].map((f) => [f, top]);
    note = "Training images of this pair.";
  } else {
    const pick = (test) => meta.train_comp_ids.filter((k) => test(meta.comp_to_crop_disease[k])).flatMap((k) => meta.gallery[k].slice(0, 1).map((f) => [f, k]));
    items = [...pick(([cc, dd]) => dd === d && cc !== c).slice(0, 2), ...pick(([cc, dd]) => cc === c && dd !== d).slice(0, 2)];
    note = `This pair has no training images. Shown: ${nice(meta.disease[d])} on other crops, and other ${nice(meta.crop[c])} images.`;
  }
  $("galNote").textContent = note;
  $("gallery").innerHTML = items.map(([f, k]) => `<figure><img src="${f}" alt="Training image: ${nice(meta.comp[k])}" loading="lazy"><figcaption>${nice(meta.comp[k])}</figcaption></figure>`).join("");
}

function colormap(t) {
  const f = (k) => Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - k))) * 255;
  return [f(3), f(2), f(1)];
}
function drawView() {
  if (!last) return;
  const view = $("view"), ctx = view.getContext("2d");
  ctx.drawImage(last.canvas, 0, 0, view.width, view.height);
  if (!$("camOn").checked || !last.att) return;
  const g = Math.round(Math.sqrt(last.att.length)), max = Math.max(...last.att) || 1;
  const small = document.createElement("canvas"); small.width = small.height = g;
  const sctx = small.getContext("2d"), id = sctx.createImageData(g, g);
  last.att.forEach((v, i) => id.data.set([...colormap(v / max), 255], i * 4));
  sctx.putImageData(id, 0, 0);
  ctx.globalAlpha = $("camAlpha").value / 100;
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(small, 0, 0, view.width, view.height);
  ctx.globalAlpha = 1;
}
$("camOn").addEventListener("change", drawView);
$("camAlpha").addEventListener("input", drawView);

// ---------------------------------------------------------------- evaluation
function renderEval() {
  const cols = [["Seen T1", (r) => r.top1.seen], ["Unseen T1", (r) => r.top1.unseen], ["H T1", (r) => r.top1.H],
    ["Seen T5", (r) => r.top5.seen], ["Unseen T5", (r) => r.top5.unseen], ["H T5", (r) => r.top5.H], ["Disease alone, unseen", (r) => r.disease_unseen]];
  $("evalTable").innerHTML = `<thead><tr><th>Model</th>${cols.map(([h]) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
    ev.rows.map((r) => `<tr class="${r.ours ? "ours" : ""}"><td>${r.name}</td>${cols.map(([, f]) => `<td>${f(r).toFixed(1)}</td>`).join("")}</tr>`).join("") + "</tbody>";
}

let cmState = null;
const BLUES = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
  .map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
function blues(t) {
  const x = Math.min(1, Math.max(0, t)) * (BLUES.length - 1), i = Math.min(Math.floor(x), BLUES.length - 2), f = x - i;
  return BLUES[i].map((v, k) => Math.round(v + (BLUES[i + 1][k] - v) * f));
}
function renderCM() {
  const split = $("cmSplit").value, cm = ev.cm[split], names = meta.comp;
  const v = ev.rows.find((r) => r.ours), key = split === "test_seen" ? "seen" : "unseen";
  $("cmMetrics").innerHTML = `<span><b>${pct(v.top1[key])}</b> Top-1</span><span><b>${pct(v.top5[key])}</b> Top-5</span><span><b>${ev.n[split]}</b> images</span>`;
  const ids = [...new Set(cm.flatMap(([t, p]) => [t, p]))].sort((a, b) => a - b);
  const idx = new Map(ids.map((id, i) => [id, i]));
  const rowSum = new Map();
  for (const [t, , c] of cm) rowSum.set(t, (rowSum.get(t) || 0) + c);
  cmState = { ids, rowSum, names, cells: new Map(cm.map(([t, p, c]) => [`${t},${p}`, c])) };
  const cv = $("cm"), ctx = cv.getContext("2d"), n = ids.length, W = cv.width, cs = W / n;
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, W);
  const norm = $("cmNorm").checked, maxC = Math.max(...cm.map(([, , c]) => c));
  $("cmMax").textContent = norm ? "100% of row" : `${maxC} images`;
  for (const [t, p, c] of cm) {
    const val = norm ? c / rowSum.get(t) : c / maxC;
    // ponytail: non-zero cells start at 15% so single errors stay visible in a 333x333 grid.
    ctx.fillStyle = `rgb(${blues(0.15 + 0.85 * val)})`;
    ctx.fillRect(idx.get(p) * cs, idx.get(t) * cs, Math.max(cs, 1.5), Math.max(cs, 1.5));
  }
  if (n <= 60) {
    ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 1; i < n; i++) { const x = Math.round(i * cs) + 0.5; ctx.moveTo(x, 0); ctx.lineTo(x, W); ctx.moveTo(0, x); ctx.lineTo(W, x); }
    ctx.stroke();
  }
  $("cmTop").innerHTML = cm.filter(([t, p]) => t !== p).sort((a, b) => b[2] - a[2]).slice(0, 10)
    .map(([t, p, c]) => `<li>${nice(names[t])} → ${nice(names[p])} <span>×${c} (${(100 * c / rowSum.get(t)).toFixed(0)}% of row)</span></li>`).join("") || "<li>No confusions.</li>";
}
["cmSplit", "cmNorm"].forEach((id) => $(id).addEventListener("change", () => ev && renderCM()));
$("cm").addEventListener("mousemove", (e) => {
  if (!cmState) return;
  const rect = $("cm").getBoundingClientRect(), n = cmState.ids.length;
  const col = Math.floor(((e.clientX - rect.left) / rect.width) * n), row = Math.floor(((e.clientY - rect.top) / rect.height) * n);
  if (col < 0 || row < 0 || col >= n || row >= n) return ($("cmTip").hidden = true);
  const t = cmState.ids[row], p = cmState.ids[col], c = cmState.cells.get(`${t},${p}`) || 0, rs = cmState.rowSum.get(t) || 0;
  const tip = $("cmTip");
  tip.innerHTML = `True: <b>${nice(cmState.names[t])}</b><br>Pred: <b>${nice(cmState.names[p])}</b><br>${c} image(s)${rs ? ` · ${(100 * c / rs).toFixed(0)}% of row (${rs})` : ""}`;
  tip.hidden = false;
  tip.style.left = `${Math.min(e.clientX - rect.left + 14, rect.width - tip.offsetWidth)}px`;
  tip.style.top = `${e.clientY - rect.top + 14}px`;
});
$("cm").addEventListener("mouseleave", () => ($("cmTip").hidden = true));

init().catch((e) => fail(`Could not load the page data: ${e.message}`));
