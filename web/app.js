// app.js - In-browser validator for the DPD-Small baselines.
//
// Flow:  photo -> plain resize 224x224 -> ImageNet normalise
//        -> 1 ONNX model (multitask) or 2 (independent crop + disease)
//        -> crop_probs, disease_probs (+ CAMs)
//        -> pair score[c] = log p(crop[c]) + log p(disease[c])  for all 333 pairs
// The same pair rule is used for both approaches, so they compare fairly.

const $ = (id) => document.getElementById(id);
let meta, summary, files;           // meta.json, summary.json, models.json
let option = null;                  // "Independent/ResNet50" etc.
let backend = "";
let lastImg = null;                 // { img, truth } to re-run on model change
let last = null;                    // { crop, cams, ch, cw, top }
const sessions = new Map();         // stem -> InferenceSession (kept, so switching back is instant)
const pct = (v) => (v == null ? "–" : `${v.toFixed(1)}%`);
const nice = (s) => s.replaceAll("_", " ");

// ---------------------------------------------------------------- loading
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get("Content-Length") || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress(got / total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

function status(text, frac = null) {
  $("status").classList.toggle("done", text == null);
  if (text != null) $("statusText").textContent = text;
  if (frac != null) $("loadBar").style.width = `${Math.round(frac * 100)}%`;
}

async function getSession(stem) {
  if (sessions.has(stem)) return sessions.get(stem);
  const f = files[stem];
  const bytes = await fetchWithProgress(f.file, (x) =>
    status(`Downloading ${stem}… ${Math.round(x * f.mb)} / ${f.mb} MB`, x));
  status(`Starting ${stem}…`);
  for (const ep of backend ? [backend] : ["webgpu", "wasm"]) {
    if (ep === "webgpu" && !navigator.gpu) continue;
    try {
      const s = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
      backend = ep;
      sessions.set(stem, s);
      return s;
    } catch (e) { console.warn(`${ep} failed, trying next backend`, e); }
  }
  throw new Error("No ONNX Runtime backend could start.");
}

async function init() {
  try {
    [meta, summary, files] = await Promise.all(
      ["meta.json", "summary.json", "models.json"].map((u) => fetch(u).then((r) => r.json())));
    files = files.files;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    renderExamples();
    renderCompare();
    const fromHash = decodeURIComponent(location.hash.slice(1));
    await selectOption(meta.options[fromHash] ? fromHash : Object.keys(meta.options)[0]);
  } catch (e) {
    status(`Could not load: ${e.message}`);
    console.error(e);
  }
}

// ---------------------------------------------------------------- model selection
async function selectOption(opt) {
  option = opt;
  const [a, b] = opt.split("/");
  $("approach").value = a; $("backbone").value = b;
  history.replaceState(null, "", `#${encodeURIComponent(opt)}`);
  for (const tr of $("cmp").querySelectorAll("tbody tr")) tr.classList.toggle("sel", tr.dataset.opt === opt);
  $("predModel").textContent = `· ${opt}`;
  renderStats();
  try {
    for (const s of meta.options[opt]) await getSession(s);
    status(null);
    if (lastImg && opt === option) classify(lastImg.img, lastImg.truth);
  } catch (e) { status(`Could not load the model: ${e.message}`); console.error(e); }
}
const onPick = () => selectOption(`${$("approach").value}/${$("backbone").value}`);
$("approach").addEventListener("change", onPick);
$("backbone").addEventListener("change", onPick);

function renderStats() {
  const s = summary[option];
  const sp = s.test_seen, su = s.test_unseen;
  const files_ = meta.options[option].map((k) => files[k]);
  $("stats").innerHTML = [
    ["Pair top-1 seen", pct(sp.pair.top1)],
    ["Pair top-1 unseen", pct(su.pair.top1)],
    ["H top-1", pct(s.harmonic.top1)],
    ["H top-5", pct(s.harmonic.top5)],
    ["Crop acc seen / unseen", `${pct(sp.crop.top1)} / ${pct(su.crop.top1)}`],
    ["Disease acc seen / unseen", `${pct(sp.disease.top1)} / ${pct(su.disease.top1)}`],
    ["Model size", `${files_.reduce((t, f) => t + f.mb, 0).toFixed(1)} MB`],
  ].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
}

// ---------------------------------------------------------------- compare table
function renderCompare() {
  const cols = [
    ["Seen T1", (s) => s.test_seen.pair.top1], ["Unseen T1", (s) => s.test_unseen.pair.top1], ["H T1", (s) => s.harmonic.top1],
    ["Seen T5", (s) => s.test_seen.pair.top5], ["Unseen T5", (s) => s.test_unseen.pair.top5], ["H T5", (s) => s.harmonic.top5],
    ["Crop seen", (s) => s.test_seen.crop.top1], ["Crop unseen", (s) => s.test_unseen.crop.top1],
    ["Disease seen", (s) => s.test_seen.disease.top1], ["Disease unseen", (s) => s.test_unseen.disease.top1],
  ];
  const opts = Object.keys(meta.options);
  const best = cols.map(([, f]) => Math.max(...opts.map((o) => f(summary[o]))));
  $("cmp").innerHTML = `<thead><tr><th>Model</th>${cols.map(([h]) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
    opts.map((o) => `<tr data-opt="${o}"><td>${o.replace("/", " · ")}</td>${cols.map(([, f], i) => {
      const v = f(summary[o]);
      return `<td class="${v === best[i] ? "best" : ""}">${v.toFixed(1)}</td>`;
    }).join("")}</tr>`).join("") + "</tbody>";
  for (const tr of $("cmp").querySelectorAll("tbody tr")) tr.onclick = () => selectOption(tr.dataset.opt);
}

// ---------------------------------------------------------------- examples + input
function renderExamples() {
  for (const ex of meta.examples) {
    const b = document.createElement("button");
    b.title = `${nice(meta.comp[ex.comp])} (${ex.split})`;
    b.innerHTML = `<img src="${ex.file}" alt="Test image: ${nice(meta.comp[ex.comp])}" loading="lazy">`;
    b.onclick = () => loadImage(ex.file, ex.comp);
    $(ex.split === "test_seen" ? "exSeen" : "exUnseen").appendChild(b);
  }
}
function loadImage(src, truth = null) {
  const img = new Image();
  img.onload = () => { lastImg = { img, truth }; classify(img, truth); };
  img.onerror = () => alert("This file could not be opened as an image.");
  img.src = src;
}
function loadFile(file) {
  if (file && file.type.startsWith("image/")) loadImage(URL.createObjectURL(file));
}
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

// Plain resize to 224x224 (no crop): the dataset was built this way, and eval uses it as-is.
function preprocess(img) {
  const S = meta.img_size;
  const c = document.createElement("canvas");
  c.width = S; c.height = S;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, S, S);
  const px = ctx.getImageData(0, 0, S, S).data;
  const data = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++)
    for (let ch = 0; ch < 3; ch++) data[ch * S * S + i] = (px[i * 4 + ch] / 255 - meta.mean[ch]) / meta.std[ch];
  return { tensor: new ort.Tensor("float32", data, [1, 3, S, S]), crop: c };
}

// ---------------------------------------------------------------- inference
const argsort = (a) => [...a.keys()].sort((i, j) => a[j] - a[i]);

async function classify(img, truth) {
  const stems = meta.options[option];
  if (!stems.every((s) => sessions.has(s))) return;     // selectOption re-runs it once loaded
  const { tensor, crop } = preprocess(img);
  const t0 = performance.now();
  const out = {};
  for (const s of stems) Object.assign(out, await sessions.get(s).run({ input: tensor }));
  const ms = performance.now() - t0;

  const pc = out.crop_probs.data, pd = out.disease_probs.data;
  const score = meta.comp_to_crop_disease.map(([c, d]) => Math.log(pc[c] + 1e-9) + Math.log(pd[d] + 1e-9));
  const mx = Math.max(...score);
  const e = score.map((v) => Math.exp(v - mx));
  const z = e.reduce((a, b) => a + b, 0);
  const pp = e.map((v) => v / z);                         // pair scores renormalised over 333 pairs, for display

  const top = { pair: argsort(pp)[0], crop: argsort(pc)[0], disease: argsort(pd)[0] };
  const [, , ch, cw] = out.crop_cam.dims;
  const cams = {
    crop: out.crop_cam.data.slice(top.crop * ch * cw, (top.crop + 1) * ch * cw),
    disease: out.disease_cam.data.slice(top.disease * ch * cw, (top.disease + 1) * ch * cw),
  };
  last = { crop, cams, ch, cw };
  showResult({ pp, pc, pd }, truth, ms);
  drawView();
}

// ---------------------------------------------------------------- output
function bars(el, probs, names, trueId, n = 5) {
  el.innerHTML = argsort(probs).slice(0, n).map((i) => `
    <li class="${i === trueId ? "truth" : ""}"><div>
      <div class="row"><span>${nice(names[i])}${names === meta.comp && meta.unseen_comp_ids.includes(i) ? ' <span class="tag unseen">unseen</span>' : ""}</span><span>${(probs[i] * 100).toFixed(1)}%</span></div>
      <div class="track"><div class="fill" style="width:${(probs[i] * 100).toFixed(1)}%"></div></div>
    </div></li>`).join("");
}

function showResult({ pp, pc, pd }, truth, ms) {
  $("empty").hidden = true; $("result").hidden = false; $("previewBox").hidden = false;
  const top = argsort(pp)[0];
  $("top1Name").textContent = nice(meta.comp[top]);
  $("top1Conf").textContent = `${(pp[top] * 100).toFixed(1)}% (pair score, normalised)`;
  $("lowConf").hidden = pp[top] >= 0.4;
  const [tc, td] = truth == null ? [null, null] : meta.comp_to_crop_disease[truth];
  if (truth == null) $("truth").innerHTML = "";
  else {
    const tag = meta.unseen_comp_ids.includes(truth) ? '<span class="tag unseen">unseen pair</span>' : '<span class="tag seen">seen pair</span>';
    $("truth").innerHTML = (truth === top ? `<span class="ok">✓ Correct</span> ` : `<span class="bad">✗ True: ${nice(meta.comp[truth])}</span> `) + tag;
  }
  bars($("topPair"), pp, meta.comp, truth);
  bars($("topCrop"), pc, meta.crop, tc);
  bars($("topDisease"), pd, meta.disease, td);
  $("timing").textContent = `Inference ${ms.toFixed(0)} ms on ${backend === "webgpu" ? "WebGPU" : "WebAssembly (CPU)"} · ${meta.options[option].length} ONNX model(s) · nothing was uploaded.`;
}

function colormap(t) {
  const f = (k) => Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - k))) * 255;
  return [f(3), f(2), f(1)];
}
function drawView() {
  if (!last) return;
  const view = $("view"), ctx = view.getContext("2d");
  ctx.drawImage(last.crop, 0, 0, view.width, view.height);
  if (!$("camOn").checked) return;
  const cam = last.cams[$("camHead").value], { ch, cw } = last;
  const pos = Array.from(cam, (v) => Math.max(0, v));
  const max = Math.max(...pos) || 1;
  const small = document.createElement("canvas");
  small.width = cw; small.height = ch;
  const sctx = small.getContext("2d");
  const id = sctx.createImageData(cw, ch);
  pos.forEach((v, i) => id.data.set([...colormap(v / max), 255], i * 4));
  sctx.putImageData(id, 0, 0);
  ctx.globalAlpha = $("camAlpha").value / 100;
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(small, 0, 0, view.width, view.height);
  ctx.globalAlpha = 1;
}
["camOn", "camHead"].forEach((id) => $(id).addEventListener("change", drawView));
$("camAlpha").addEventListener("input", drawView);

init();
