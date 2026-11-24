// build_report.js - builds report/COS30082_Group13_Report.docx from the numbers in the repo.
// Run: NODE_PATH=<dir with docx installed>/node_modules node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
  WidthType, ShadingType, BorderStyle, LevelFormat, Footer, PageNumber, ExternalHyperlink,
} = require("docx");

const FIG = path.join(__dirname, "figs");
const W = 9638;                       // text width in DXA: A4 (11906) - 2 x 1134 margins
const FONT = "Calibri";

// ---------------------------------------------------------------- helpers
const runs = (text) => {               // **bold** and *italic* inline markup
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/)) {
    if (!part) continue;
    if (part.startsWith("**")) out.push(new TextRun({ text: part.slice(2, -2), bold: true }));
    else if (part.startsWith("*")) out.push(new TextRun({ text: part.slice(1, -1), italics: true }));
    else out.push(new TextRun(part));
  }
  return out;
};
const P = (text, o = {}) => new Paragraph({ children: runs(text), spacing: { after: 100 }, alignment: AlignmentType.JUSTIFIED, ...o });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const B = (text) => new Paragraph({ children: runs(text), numbering: { reference: "bul", level: 0 }, spacing: { after: 40 }, alignment: AlignmentType.JUSTIFIED });
const EQ = (text) => new Paragraph({ children: [new TextRun({ text, italics: true })], alignment: AlignmentType.CENTER, spacing: { before: 60, after: 100 } });
const CAP = (text) => new Paragraph({ children: runs(text).map((r) => r), alignment: AlignmentType.CENTER, spacing: { before: 40, after: 160 }, style: "Caption" });
const IMG = (file, w, h) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80, after: 0 }, keepNext: true,
  children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIG, file)), transformation: { width: w, height: h },
    altText: { title: file, description: file, name: file } })],
});

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
// rows: array of arrays of strings; first row = header. widths in DXA, must sum to W (or table width).
function TABLE(rows, widths, { hl = [], small = false, align = [] } = {}) {
  const size = small ? 17 : 18;
  const tw = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: tw, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0, cantSplit: true,
      children: r.map((c, j) => new TableCell({
        borders, width: { size: widths[j], type: WidthType.DXA },
        shading: i === 0 ? { fill: "E8EEF7", type: ShadingType.CLEAR, color: "auto" }
          : hl.includes(i) ? { fill: "E6F4EA", type: ShadingType.CLEAR, color: "auto" } : undefined,
        margins: { top: 30, bottom: 30, left: 70, right: 70 },
        children: [new Paragraph({
          alignment: (align[j] ?? (j === 0 ? "L" : "C")) === "L" ? AlignmentType.LEFT : AlignmentType.CENTER,
          children: [new TextRun({ text: c, size, bold: i === 0 || hl.includes(i) })],
        })],
      })),
    })),
  });
}
const TCAP = (text) => new Paragraph({ children: runs(text), alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, style: "Caption", keepNext: true });

// ---------------------------------------------------------------- symptom sources (appendix)
const SRC = (() => {   // parse S{n}: ("title", "url") from 3_Final_New_Approach/training/symptoms_sources.py
  const py = fs.readFileSync(path.join(__dirname, "..", "3_Final_New_Approach", "training", "symptoms_sources.py"), "utf8");
  return [...py.matchAll(/"(S\d+)": \("([^"]+)", "([^"]+)"\)/g)].map((m) => [m[1], m[2], m[3]]);
})();

// ---------------------------------------------------------------- content
const title = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "COS30082 Applied Machine Learning — Group 13", size: 22, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Out-of-Distribution Plant Disease Identification", size: 34, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: "Enoch Eren Chua (104401704) · Terence Chin Seng Wong (104404059) · Samuel Hao Jun Chang (102775957) · Ji Yan Lim (104382537)", size: 19 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [
    new TextRun({ text: "Code: ", size: 18, bold: true }), new ExternalHyperlink({ link: "https://github.com/z4pyyy/COS30082_Group13_AML", children: [new TextRun({ text: "github.com/z4pyyy/COS30082_Group13_AML", size: 18, style: "Hyperlink" })] }),
    new TextRun({ text: "   ·   GUI: ", size: 18, bold: true }), new ExternalHyperlink({ link: "https://z4pyyy-plantdiseasevalidator.static.hf.space", children: [new TextRun({ text: "z4pyyy-plantdiseasevalidator.static.hf.space", size: 18, style: "Hyperlink" })] }),
  ] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [
    new TextRun({ text: "Video (≤ 8 min): ", size: 18, bold: true }), new TextRun({ text: "[LINK — TODO]", size: 18, highlight: "yellow" }),
    new TextRun({ text: "   ·   Read-only repo access given to shlee@swinburne.edu.my on ", size: 18 }), new TextRun({ text: "[DATE — TODO]", size: 18, highlight: "yellow" }),
  ] }),
];

const body = [
  // ======================================================== 1
  H1("1. Introduction"),
  P("Plant-disease classifiers are usually trained under a closed-world assumption: every category seen at test time also appears in training. In the field, new crop–disease combinations appear with no labelled images, and such models then force them into a known category. We write the full category set as *C = S ∪ U*, where *S* are the categories seen in training and *U* the unseen ones. We treat each category as a crop–disease pair. An unseen pair combines a crop and a disease that were both seen in training, but never together. We also consider robustness, because field photos differ from laboratory photos in lighting, background, viewpoint and quality. All experiments use DPD-Small (55 crops, 175 diseases, 323 seen and 10 unseen pairs) and are scored with Top-1/Top-5 accuracy, the harmonic mean H of seen and unseen accuracy, and confusion matrices."),
  P("Our contributions are:"),
  B("**Baselines under one protocol.** Independent crop and disease CNNs, and one multitask CNN, each with three backbones, trained and scored with the same pipeline and the same pair-scoring rule."),
  B("**Proposed approach: DINO-CPS (DINOv2 Compositional Pair Scoring).** Two fully fine-tuned, self-supervised DINOv2 ViT-L models — a crop model and a disease model with attention pooling over lesion patches — combined by a fixed compositional rule, fused with the CNN score and calibrated on validation only. A six-component ablation (adversarial feature disentanglement, synthetic compositional augmentation, field-domain adaptation and vision-language zero-shot transfer) tests what improves generalisation and robustness."),
  B("**In-browser GUI.** The final model runs entirely in the browser (int8 ONNX) and shows the predicted pair, the disease model's attention, the matching training images and the evaluation required by the brief."),

  // ======================================================== 2
  H1("2. Dataset and Preprocessing"),
  P("**DPD-Small** contains 11,448 images: 8,000 for training, 2,000 seen test images (*test_seen*) and 1,448 unseen test images (*test_unseen*). The 333 pairs comprise 323 seen and 10 unseen pairs; each unseen pair uses a crop and a disease that appear in training through other pairs."),
  P("**Cleaning (DPD_Small_clean).** We removed 5 training images that were near-duplicates of *test_seen* images, re-saved 11 files with wrong extensions, converted 28 RGBA images to RGB on a white background, and flagged 349 images smaller than 224 px (upscaled). Every image is resized to 224×224 without cropping, so lesions near the leaf edge are kept. The test sets are otherwise unchanged."),
  P("**Splits for model selection.** The 7,995 remaining training images form *train* (6,995), *val* (773, seen pairs; 10% per pair) and *val_unseen* (227 images of 6 seen pairs held out of *train*). *val_unseen* is a proxy for unseen pairs. The two test sets are never used for tuning, early stopping or model choice."),
  P("**Data characteristics.** The distribution is long-tailed (Figure 1): 1 to 544 images per pair, median 7. One source (CDDM) provides 38% of training images. Field photos are only 6% of training but 27% of *test_unseen*, so robustness to field conditions matters for unseen-pair accuracy."),
  IMG("fig_pairs.png", 520, 179),
  CAP("Figure 1. Training images per crop–disease pair (log scale), train + val."),

  // ======================================================== 3
  H1("3. Methodology"),
  H2("3.1 Overall framework"),
  P("All models follow one compositional framework (Figure 2). A crop classifier gives *p(crop)* over 55 classes and a disease classifier gives *p(disease)* over 175 classes. Each of the 333 pairs is then scored by"),
  EQ("score(c, d) = log p(crop = c) + log p(disease = d)"),
  P("and the pairs are ranked for Top-1/Top-5. An unseen pair can be predicted because its crop and its disease were each learned separately. The same rule is used for every model, so differences come from the learned features and not from the scoring."),
  IMG("fig_baseline.png", 600, 142),
  CAP("Figure 2. Baseline framework. Multitask models replace the two CNNs by one shared backbone with two heads."),

  H2("3.2 Baseline models"),
  P("**Independent CNNs** train two separate networks: a crop classifier and a disease classifier. **Multitask CNN** uses one shared backbone with two linear heads and the loss *L = CE(crop) + CE(disease)*. Both use three ImageNet-1k backbones from timm — ResNet50, EfficientNet-B0 and ConvNeXt-Tiny — fully fine-tuned with cross-entropy and label smoothing 0.1. The protocol is identical for both approaches (Table 1), so the only difference is \"one network or two\"."),
  TCAP("Table 1. Shared baseline training settings."),
  TABLE([
    ["Setting", "Value"],
    ["Optimiser / schedule", "AdamW (weight decay 0.05); OneCycleLR with 10% warm-up"],
    ["Learning rate", "ResNet50 3e-4 · EfficientNet-B0 5e-4 · ConvNeXt-Tiny 1e-4"],
    ["Batch / epochs", "64 (effective); 30 max, early-stopping patience 10"],
    ["Augmentation", "RandomResizedCrop(224, scale 0.6–1.0), H/V flips, rotation ±20°, ColorJitter(0.2), RandomErasing(p 0.25)"],
    ["Model selection", "best epoch = highest H of val and val_unseen Top-1"],
    ["Seed / hardware", "42 / NVIDIA Tesla T4 (Google Colab, Kaggle)"],
  ], [2400, 7238], { align: ["L", "L"] }),

  H2("3.3 Proposed approach: DINO-CPS"),
  P("**Motivation.** The baselines recognise the crop well on unseen pairs (62–70%) but the disease poorly (5–14%; Section 5). An earlier prototype (frozen DINOv2-L and BioCLIP-2 features with linear probes, fused with the CNN) raised H to 62.0, but its disease model alone was correct on only 4.6% of unseen-pair images. It had learned a shortcut: \"this crop usually has this disease\". DINO-CPS aims to make the disease feature depend on the lesion, not on the crop."),
  P("**Architecture (Figure 3).** Each classifier is a DINOv2 ViT-L/14 [1], pre-trained without labels on 142M images, fully fine-tuned on DPD-Small. A 224×224 image becomes 256 patch tokens plus one CLS token (1,024-d). The crop model uses the CLS token and a linear head (55). The disease model uses gated attention pooling [2] over the 256 patch tokens,"),
  EQ("aᵢ = softmax( wᵀ [ tanh(V tᵢ) ⊙ σ(U tᵢ) ] ),   z = Σᵢ aᵢ tᵢ"),
  P("followed by a linear head (175). The weights *aᵢ* need no lesion labels; Figure 4 shows that they concentrate on lesions. The final pair score adds the team ConvNeXt-T pair score and calibrated stacking [3]:"),
  EQ("s(c, d) = log p₍dino₎(c) + log p₍dino₎(d) + w·[log p₍cnn₎(c) + log p₍cnn₎(d)] − b·𝟙[(c, d) ∈ S] + q·prior(c, d)"),
  P("The bias *b* lowers the 317 pairs seen in training so that unseen pairs can win. *prior* is a host-range prior from botanical family (+1 if the disease is seen on a crop of the same family, −1 for narrow-host diseases). We added calibration only after inspecting the raw predictions: without it, 82% of the CNN's and 71–74% of DINO-CPS's *val_unseen* images were assigned to a seen pair (Table 5). *w*, *b* and *q* are tuned on *val* + *val_unseen* only."),
  IMG("fig_dino_cps_arch.png", 640, 242),
  CAP("Figure 3. DINO-CPS architecture (final model). Steps 1–3 are learned on train only; epoch, w and b are chosen on validation; test is scored once."),
  P("**Generalisation components (ablation).** Each run adds one component to the previous one (3 seeds each):"),
  B("**Attn — attention pooling** (above): focuses the disease feature on lesion patches."),
  B("**Adv — adversarial feature disentanglement:** a crop classifier on the disease feature, trained through a gradient-reversal layer [4] (weight 0.3, DANN ramp), so the disease feature carries no crop information. Applied only to diseases seen on ≥ 2 crops."),
  B("**Syn — synthetic compositional augmentation:** 4,000 images where lesion pixels (hue-based mask) are cut from a diseased leaf and pasted onto a healthy leaf of a plausible new host crop [5]; this creates crop–disease combinations absent from training."),
  B("**SymText — vision-language zero-shot transfer:** BioCLIP-2 [6] image embeddings matched by cosine similarity to text embeddings of each disease (zero-shot learning with semantic embeddings). Prompts contain no crop word: three templates per disease plus a symptom description for 72 diseases (e.g. *rust: small orange to reddish-brown powdery pustules, mostly on the leaf underside*). The text embeddings are fixed; the last 6 vision blocks are fine-tuned with cross-entropy over the 175 disease texts. Every description was checked against an extension or APS source (Appendix A: 62 supported, 9 partly supported, 1 unsupported)."),
  P("**Field — robustness to field images.** The crop model (all runs) and the disease model in *DINO-CPS+Adv+Syn+Field* use field-style augmentation (crops at scale 0.3–1.0, strong colour jitter, Gaussian blur, perspective, JPEG compression 25–90) and oversample field photos to 25% of each epoch. That run also adds a lab/field domain adversary through gradient reversal (weight 0.1), a form of adversarial domain adaptation."),
  TCAP("Table 2. DINO-CPS training settings (differences from Table 1)."),
  TABLE([
    ["Setting", "Value"],
    ["Backbone", "DINOv2 ViT-L/14 (vit_large_patch14_dinov2.lvd142m), 303M parameters per model"],
    ["Optimiser / schedule", "AdamW (wd 0.05), lr 1e-5 backbone / 1e-3 heads; linear warm-up then cosine"],
    ["Batch / epochs / precision", "48 / 15 / bf16, drop-path 0.1"],
    ["Loss", "CE + label smoothing 0.1 (+ 0.3·CE_adv crop, + 0.1·CE_adv domain where used)"],
    ["VL model (SymText)", "BioCLIP-2, last 6 blocks, lr 1e-5, 10 epochs, frozen text embeddings"],
    ["Selection / seeds / hardware", "best epoch by val H; seeds 0, 1, 2; NVIDIA RTX 5090"],
  ], [2700, 6938], { align: ["L", "L"] }),

  // ======================================================== 4
  H1("4. Experimental Setup"),
  P("Following the brief, we report Top-1 and Top-5 pair accuracy on *test_seen* and *test_unseen*, the harmonic mean *H = 2·S·U / (S + U)*, and the confusion matrix on *test_seen*. We also report crop- and disease-head accuracy separately; disease accuracy on *test_unseen* shows whether a model recognises the disease or only the crop. Baseline numbers come from the exported fp16 ONNX models and match the PyTorch evaluation within 0.15 pp. DINO-CPS numbers are the mean ± standard deviation over 3 seeds. Every choice (epoch, *w*, *b*, *q*, fusion weights) uses validation only; the test sets are scored once."),

  // ======================================================== 5
  H1("5. Results and Discussion"),
  H2("5.1 Main results"),
  TCAP("Table 3. Pair accuracy (%) on the test sets. Calibrated rows use the bias b tuned on validation."),
  TABLE([
    ["Model", "Seen T1", "Unseen T1", "H T1", "Seen T5", "Unseen T5", "H T5"],
    ["Independent ResNet50", "63.9", "8.4", "14.9", "84.7", "40.9", "55.1"],
    ["Independent EfficientNet-B0", "64.0", "7.9", "14.0", "85.2", "37.8", "52.3"],
    ["Independent ConvNeXt-Tiny", "63.8", "13.6", "22.4", "83.8", "39.9", "54.0"],
    ["Multitask ResNet50", "61.8", "4.1", "7.6", "85.3", "31.7", "46.2"],
    ["Multitask EfficientNet-B0", "61.0", "1.4", "2.7", "83.3", "22.3", "35.2"],
    ["Multitask ConvNeXt-Tiny", "63.3", "4.2", "7.9", "84.0", "20.9", "33.5"],
    ["Indep. ConvNeXt-Tiny + calibration", "54.3", "56.6", "55.4", "79.4", "86.7", "82.9"],
    ["DINO-CPS (proposed), 3 seeds", "63.8 ± 1.7", "76.1 ± 1.8", "69.4 ± 1.3", "89.2 ± 0.8", "97.3 ± 0.2", "93.1 ± 0.4"],
  ], [3238, 1050, 1100, 1050, 1050, 1100, 1050], { hl: [8] }),
  TCAP("Table 4. Head-level accuracy (%) on the test sets."),
  TABLE([
    ["Model", "Crop seen", "Crop unseen", "Disease seen", "Disease unseen"],
    ["Independent ResNet50", "88.0", "65.3", "65.0", "8.6"],
    ["Independent EfficientNet-B0", "89.4", "67.6", "64.9", "9.0"],
    ["Independent ConvNeXt-Tiny", "89.0", "70.2", "64.3", "14.1"],
    ["Multitask ResNet50", "88.0", "67.4", "63.8", "7.7"],
    ["Multitask EfficientNet-B0", "86.5", "62.3", "63.3", "5.2"],
    ["Multitask ConvNeXt-Tiny", "88.8", "62.5", "65.6", "12.0"],
    ["DINO-CPS (proposed), 3 seeds", "94.9", "86.1", "72.2", "8.0"],
  ], [3638, 1500, 1500, 1500, 1500], { hl: [7] }),

  H2("5.2 Discussion: baselines"),
  P("**Seen vs unseen gap.** All baselines reach 61–64% Top-1 on seen pairs but only 1–14% on unseen pairs. They recognise combinations seen in training, but fail when a known disease appears on a new crop. Because the pair score adds the two log-probabilities, a low score from either head pushes the correct pair down the ranking."),
  P("**Crop vs disease.** Unseen crop accuracy is 62–70%, unseen disease accuracy only 5–14% (Table 4). Leaf shape and venation stay the same across the diseases of a crop, but a disease's appearance changes with the host, and in training each disease is tied to the crops it was photographed on. Disease recognition is therefore the bottleneck."),
  P("**Independent vs multitask.** Seen accuracy is similar, but independent models are much better on unseen pairs (H T1 14–22 vs 3–8). A likely reason is that the shared backbone lets the disease head use crop features, which helps on known pairs and hurts on new ones. We did not test this directly."),
  P("**Backbone and Top-5.** Backbones differ by at most 3 pp on seen pairs; ConvNeXt-Tiny is best on unseen pairs in the independent track (single seed, so differences are uncertain). Unseen Top-5 (21–41%) is far above Top-1: the correct pair is often close, but the raw scores rank seen pairs above it. This motivates calibration."),

  H2("5.3 Calibration"),
  TCAP("Table 5. Raw scores (b = 0) vs calibrated scores, inspected on validation first."),
  TABLE([
    ["Model", "Val seen T1", "Val unseen T1", "Val H", "Unseen val images sent to a seen pair", "Test H T1"],
    ["Indep. ConvNeXt-T, raw", "81.4", "16.7", "27.8", "82.4%", "22.3"],
    ["Indep. ConvNeXt-T, calibrated (b = 5.51)", "70.5", "59.5", "64.5", "19.8%", "55.4"],
    ["DINO-CPS, raw (3 seeds)", "86.5–86.8", "24.7–28.6", "38.4–43.1", "71–74%", "17.5–18.6"],
    ["DINO-CPS, calibrated (b ≈ 5.6)", "77.9–80.1", "70.0–74.4", "74.5–77.2", "12–19%", "66.9–69.4"],
  ], [3038, 1050, 1150, 1050, 2200, 1150]),
  P("Raw scores send most unseen images to a seen pair: training only ever rewarded seen pairs, the known seen-class bias of generalised zero-shot learning. One constant *b*, tuned on validation, removes most of this bias. It costs seen accuracy (81.4 → 70.5 on validation) but raises H from 22.4 to 55.4 on test for the same CNN. We therefore compare DINO-CPS with the calibrated baseline (55.4), not with the raw one."),

  H2("5.4 Proposed approach vs baselines, and ablation"),
  TCAP("Table 6. DINO-CPS ablation, final variant (+ prior + CNN fusion), mean of 3 seeds. Selection uses val H only."),
  TABLE([
    ["Run", "Adds", "val H", "Test H T1", "Field T1", "Disease alone, unseen"],
    ["DINO-CPS-Base", "DINOv2 ViT-L, CLS head", "75.1", "68.2 ± 2.2", "38.6", "8.7"],
    ["DINO-CPS (final)", "+ lesion attention", "76.6", "69.4 ± 1.3", "43.0", "8.0"],
    ["DINO-CPS+Adv", "+ crop adversary", "75.5", "65.4 ± 3.4", "40.1", "8.7"],
    ["DINO-CPS+Adv+Syn", "+ synthetic pairs", "73.4", "66.8 ± 1.8", "39.8", "14.5"],
    ["DINO-CPS+Adv+Syn+Field", "+ field adaptation", "71.8", "64.9 ± 1.8", "40.2", "11.0"],
    ["SymText", "BioCLIP-2 + symptom text", "71.2", "64.4 ± 0.8", "38.2", "13.4"],
    ["DINO-CPS+Adv+Syn ⊕ SymText", "fused", "69.6", "65.5 ± 2.3", "36.6", "15.2"],
  ], [3100, 1938, 1000, 1300, 1000, 1300], { hl: [2] }),
  P("**Against the baselines.** DINO-CPS raises H Top-1 from 55.4 (calibrated baseline) to 69.4 ± 1.3 and H Top-5 from 82.9 to 93.1. Seen Top-1 also recovers to 63.8%, the level of the uncalibrated CNNs, so calibration no longer trades away seen accuracy. Field-photo pair accuracy rises from 23.3% to 43.0% (lab: 63.2% → 75.5%), which answers the robustness question: the DINOv2 crop model trained with field augmentation is far more robust than the CNNs."),
  P("**What the ablation shows.** Attention pooling (Attn) is the only component that raised validation H, so it is the final model. The adversary (Adv), synthetic pairs (Syn), field adaptation (Field) and symptom-text transfer (SymText) lowered test H by 3–5 points, but Syn and SymText almost doubled disease-only accuracy on unseen pairs (8.0% → 14.5–15.2%). These components do make the disease model more crop-invariant, but the gain does not reach the pair score."),
  P("**An honest limitation.** The unseen pair score depends mostly on the crop model. Only 16 pairs have no training images, on 13 crops; 10 of those crops have exactly one such pair. After calibration a correct crop usually gives the correct unseen pair, so the disease model is rarely decisive. DINO-CPS's disease model alone is right on only 8.0% of unseen images (CNN: 14.1%). The shortcut is reduced by Syn/SymText but not solved; on a crop with several candidate unseen pairs, the disease model would decide and would often fail."),
  IMG("fig_attention.png", 640, 179),
  CAP("Figure 4. Disease-model attention (DINO-CPS, seed 0). Top: test_seen. Bottom: test_unseen. Weights concentrate on lesions."),

  H2("5.5 Confusion matrix (test_seen)"),
  P("Figure 5 shows the final model on *test_seen* (seed 0, 61.9% Top-1). The diagonal dominates; most errors stay within one crop: tomato Septoria leaf spot → tomato early blight (25 images, 64% of the class), paddy healthy → paddy blast (19, 59%), tomato bacterial spot → tomato early blight (17, 63%) and grape esca → grape black rot (13, 81%). These pairs share small dark spots or necrotic patches at 224 px. Several errors predict tomato early blight, which is an *unseen* pair: this is the cost of calibration, which moves borderline seen images to unseen pairs. The vertical stripes in Figure 5 are such \"attractor\" pairs."),
  IMG("fig_cm_seen.png", 300, 335),
  CAP("Figure 5. Pair confusion matrix of the final model on test_seen (333 classes, row-normalised)."),

  H2("5.6 Model selection"),
  P("We select **DINO-CPS** (attention pooling, + prior + CNN fusion). It has the highest mean validation H of all 48 model × variant combinations (76.6) and also the best test H Top-1 (69.4 ± 1.3); its H Top-5 (93.1) is within 0.2 pp of the best. Its cost is size: two 303M-parameter ViT-L models (≈ 660M parameters with the CNNs) against 28M for one ConvNeXt-Tiny. We accept this because the brief's headline metric improves by 14 H points and the model still runs on a CPU in a browser after int8 quantisation (Section 6)."),

  // ======================================================== 6
  H1("6. User Interface"),
  P("The GUI is a static Hugging Face Space (link above); the whole model runs in the user's browser with ONNX Runtime Web, so no image is uploaded. The two ViT-L models are exported to int8 ONNX (309 MB each, weight-only dynamic quantisation of MatMul/Gemm) together with the two ConvNeXt-T models. On all 4,448 validation and test images, int8 agrees with the original model's Top-1 on 97–100% of images, and accuracy changes by at most 0.3 pp. The deployed model is a seed-0 retrain with saved weights; its own test numbers (H T1 66.9, H T5 93.1) are shown on the page. After a one-time download (≈ 720 MB, cached) one prediction takes about 6 s on a laptop CPU."),
  P("Features (Figure 6): (1) upload, drag-and-drop or paste a photo, or click a seen/unseen test example; (2) the top crop–disease pair with confidence, the Top-5 pairs and the crop and disease Top-5; the disease model's attention heatmap; and, as the brief requires, **the matching training images** — for a seen pair its own training photos, for an unseen pair the same disease on other crops and other photos of the same crop; (3) a \"How it works\" section with the pipeline; and an evaluation section with the brief's metrics and an interactive pair confusion matrix."),
  IMG("fig_gui.png", 330, 411),
  CAP("Figure 6. GUI: an unseen pair (pepper bell bacterial spot) predicted correctly, with matching training images and attention."),

  // ======================================================== 7
  H1("7. Conclusion"),
  P("The CNN baselines learn seen pairs but fail on unseen ones (H ≤ 22.4), because the disease head does not transfer across crops; calibrating the seen-pair bias alone lifts the best CNN to 55.4. Our approach, DINO-CPS — two fine-tuned self-supervised DINOv2 models with attention pooling, CNN fusion and validation-tuned calibration — reaches H 69.4 ± 1.3 (Top-5 93.1) and almost doubles accuracy on field photos. The main limitation is that the unseen-pair gain comes mostly from the crop model: crop-invariance components (adversarial disentanglement, synthetic pairs, vision-language symptom text) raise disease-only accuracy to 15% but lower H. Future work: train these components jointly with a pair-level objective, evaluate on unseen pairs with several candidate diseases per crop, and use lesion masks instead of colour heuristics for synthesis."),

  // ======================================================== refs
  H1("References"),
  ...[
    "[1] M. Oquab et al., \"DINOv2: Learning Robust Visual Features without Supervision,\" Trans. Machine Learning Research, 2024.",
    "[2] M. Ilse, J. Tomczak and M. Welling, \"Attention-based Deep Multiple Instance Learning,\" in Proc. ICML, 2018.",
    "[3] W.-L. Chao, S. Changpinyo, B. Gong and F. Sha, \"An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild,\" in Proc. ECCV, 2016.",
    "[4] Y. Ganin and V. Lempitsky, \"Unsupervised Domain Adaptation by Backpropagation,\" in Proc. ICML, 2015.",
    "[5] D. Dwibedi, I. Misra and M. Hebert, \"Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection,\" in Proc. ICCV, 2017.",
    "[6] J. Gu et al., \"BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning,\" arXiv:2505.23883, 2025.",
    "[7] K. He, X. Zhang, S. Ren and J. Sun, \"Deep Residual Learning for Image Recognition,\" in Proc. CVPR, 2016.",
    "[8] M. Tan and Q. Le, \"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,\" in Proc. ICML, 2019.",
    "[9] Z. Liu et al., \"A ConvNet for the 2020s,\" in Proc. CVPR, 2022.",
    "[10] C. Szegedy et al., \"Rethinking the Inception Architecture for Computer Vision,\" in Proc. CVPR, 2016 (label smoothing).",
    "[11] R. Wightman, \"PyTorch Image Models (timm),\" GitHub repository, 2019.",
    "[12] ONNX Runtime developers, \"ONNX Runtime,\" https://onnxruntime.ai, 2021.",
  ].map((t) => new Paragraph({ children: [new TextRun({ text: t, size: 18 })], spacing: { after: 30 } })),

  // ======================================================== appendix
  H1("Appendix A. Sources of the symptom descriptions (SymText)"),
  P("Each of the 72 descriptions in 3_Final_New_Approach/training/symptoms.json was checked against the source below (accessed 24 Nov 2026). The full mapping, with the status and a note per disease, is in 3_Final_New_Approach/training/symptoms_sources.csv. Partly supported: anthracnose, black_rot, leaf_blight, blight_leaf, cercospora_leaf_blight, bacterial_blight, scorch, ring_spot, nutrition_deficiency. Unsupported: white_spot (tea).", { style: "Small" }),
  ...SRC.map(([id, t, u]) => new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: `[${id}] ${t}. `, size: 15 }), new TextRun({ text: u, size: 15, color: "1F4E9A" })] })),
];

const doc = new Document({
  creator: "COS30082 Group 13", title: "Out-of-Distribution Plant Disease Identification",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 22, bold: true, color: "2E5597" }, paragraph: { spacing: { before: 140, after: 60 }, keepNext: true, outlineLevel: 1 } },
      { id: "Caption", name: "Caption", basedOn: "Normal", run: { font: FONT, size: 17, italics: true, color: "444444" } },
      { id: "Small", name: "Small", basedOn: "Normal", run: { font: FONT, size: 17 } },
    ],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: "777777" })] })] }) },
    children: [...title, ...body],
  }],
});

Packer.toBuffer(doc).then((b) => {
  const out = path.join(__dirname, "COS30082_Group13_Report.docx");
  fs.writeFileSync(out, b);
  console.log("wrote", out);
});
