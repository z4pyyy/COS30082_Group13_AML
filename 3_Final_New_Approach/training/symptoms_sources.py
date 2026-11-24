"""Source for every symptom description in symptoms.json (Dr Lee, option a: external descriptions must be cited).

    python3 3_Final_New_Approach/training/symptoms_sources.py     # writes symptoms_sources.csv and checks every key is covered

status: supported = the source states the same visual symptoms;
        partial   = the source supports the main symptom but not every detail (note says which);
        none      = no credible source found for this wording (note says why). Healthy needs no source.
Sources checked 2026-11-24.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

S = {  # id: (short reference, URL)
    "S1": ("Univ. Minnesota Extension, Common rust on corn", "https://extension.umn.edu/corn-pest-management/common-rust-corn"),
    "S2": ("Oklahoma State Univ. Extension, Identifying rust diseases of wheat and barley", "https://extension.okstate.edu/fact-sheets/identifying-rust-diseases-of-wheat-and-barley"),
    "S3": ("Univ. Arizona Extension, AZ1755 Stripe rust of wheat", "https://extension.arizona.edu/sites/extension.arizona.edu/files/pubs/az1755-2018.pdf"),
    "S4": ("UMass Extension, Bean rust", "https://www.umass.edu/agriculture-food-environment/vegetable/fact-sheets/bean-rust"),
    "S5": ("Univ. Minnesota Extension, Cedar-apple rust", "https://extension.umn.edu/plant-diseases/cedar-apple-rust"),
    "S6": ("UH CTAHR, Algal leaf spot (Cephaleuros)", "https://www.ctahr.hawaii.edu/uhmg/FAQ/faq-algal-lfspot.asp"),
    "S7": ("UC IPM, Pest Notes: Powdery mildew (pn7406)", "https://ipm.ucanr.edu/PMG/PESTNOTES/pn7406.html"),
    "S8": ("Ohio State Univ., Cucurbit downy mildew facts", "https://u.osu.edu/vegetablediseasefacts/?p=227"),
    "S9": ("Univ. Nebraska Extension, Anthracnose", "https://byf.unl.edu/anthracnose"),
    "S10": ("UF/IFAS EDIS RH007, Brown spot of rice", "https://ipm.ifas.ufl.edu/pdfs/RH00700.pdf"),
    "S11": ("Univ. Kentucky, Plant disease basics (PSEP)", "https://www.uky.edu/Ag/Entomology/PSEP/11pests1disease.html"),
    "S12": ("Alabama Extension, Bacterial spot of tomato and pepper", "https://www.aces.edu/blog/topics/crop-production/bacterial-spot-of-tomato-and-pepper/"),
    "S13": ("Penn State Extension, Grape disease: black rot", "https://extension.psu.edu/grape-disease-black-rot"),
    "S14": ("Univ. Illinois Extension, Black spot (rose)", "https://extension.illinois.edu/plant-problems/black-spot"),
    "S15": ("Cornell Vegetables, Alternaria leaf spot of brassicas", "https://vegetables.cornell.edu/pest-management/disease-factsheets/alternaria-leaf-spot-of-brassicas"),
    "S16": ("Univ. Minnesota Extension, Alternaria leaf blight", "https://extension.umn.edu/disease-management/alternaria-leaf-blight"),
    "S17": ("Michigan State Univ. IPM, Alternaria blotch", "https://www.canr.msu.edu/ipm/diseases/alternaria_blotch"),
    "S18": ("Oklahoma State Univ. Extension, Early blight of tomato", "https://extension.okstate.edu/programs/digital-diagnostics/plant-diseases/early-blight-of-tomato"),
    "S19": ("Utah State Univ. Extension, Late blight (tomato)", "https://extension.usu.edu/vegetableguide/tomato-pepper-eggplant/late-blight"),
    "S20": ("UF/IFAS EDIS PP351, Target spot of tomato", "https://edis.ifas.ufl.edu/publication/PP351"),
    "S21": ("Washington State Univ. HortSense, Pepper/eggplant mosaic viruses", "https://hortsense.cahnrs.wsu.edu/fact-sheet/pepper-eggplant-mosaic-viruses/"),
    "S22": ("TNAU Agritech, Yellow mosaic of greengram/blackgram", "https://agritech.tnau.ac.in/crop_protection/green_gram_disease/yellow_mosaic.html"),
    "S23": ("Penn State Extension, Peach leaf curl", "https://extension.psu.edu/peach-disease-peach-leaf-curl"),
    "S24": ("UC IPM, Tomato yellow leaf curl", "https://ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/"),
    "S25": ("Michigan State Univ. IPM, Apple scab", "https://www.canr.msu.edu/ipm/diseases/apple_scab"),
    "S26": ("APS Plant Health Instructor, Brown rot of stone fruits", "https://apsnet.org/edcenter/disandpath/fungalasco/pdlessons/Pages/BrownRotStoneFruits.aspx"),
    "S27": ("Michigan State Univ., Cercospora leaf spot of sugarbeet", "https://www.canr.msu.edu/resources/cercospora-leaf-spot-of-sugarbeet"),
    "S28": ("Crop Protection Network, Cercospora leaf blight of soybean", "https://cropprotectionnetwork.org/encyclopedia/cercospora-leaf-blight-of-soybean"),
    "S29": ("Crop Protection Network, Frogeye leaf spot of soybean", "https://cropprotectionnetwork.org/encyclopedia/frogeye-leaf-spot-of-soybean"),
    "S30": ("UC IPM, Loose smut (small grains)", "https://ipm.ucanr.edu/agriculture/small-grains/loose-smut/"),
    "S31": ("Texas A&M AgriLife, PLPA-FC010 Bacterial blight of cotton", "https://ccag.tamu.edu/wp-content/uploads/sites/10/2020/12/16_FS_FC010_Cot_Bact_Bl.pdf"),
    "S32": ("Univ. Illinois Extension, Angular leaf spot", "https://extension.illinois.edu/plant-problems/angular-leaf-spot"),
    "S33": ("Crop Protection Network, Bacterial leaf streak of corn", "https://cropprotectionnetwork.org/publications/bacterial-leaf-streak-of-corn"),
    "S34": ("Michigan State Univ., Blueberry scorch virus", "https://www.canr.msu.edu/uploads/files/Scorch.pdf"),
    "S35": ("Univ. Maryland Extension, Septoria leaf spot of tomato", "https://extension.umd.edu/resource/septoria-leaf-spot-tomatoes"),
    "S36": ("APS Plant Health Instructor, Papaya ringspot virus", "https://apsnet.org/edcenter/pdlessons/Pages/PapayaRingspotvirus.aspx"),
    "S37": ("APS Plant Health Instructor, Rice sheath blight", "https://apsnet.org/edcenter/pdlessons/Pages/RiceSheath.aspx"),
    "S38": ("UC IPM, Spider mites (pn7405)", "https://www.ipm.ucanr.edu/PMG/PESTNOTES/pn7405.html"),
    "S39": ("UC IPM, Botrytis blight or gray mold", "https://ipm.ucanr.edu/home-and-landscape/botrytis-blight-or-gray-mold/"),
    "S40": ("UF/IFAS EDIS VH045, Phytophthora blight of eggplant", "https://edis.ifas.ufl.edu/publication/VH045"),
    "S41": ("UF/IFAS U-Scout, Fusarium wilt of tomato", "https://plantpath.ifas.ufl.edu/u-scout/tomato/fusarium-wilt.html"),
    "S42": ("Univ. Illinois Extension, Bacterial wilt of cucurbits", "https://extension.illinois.edu/plant-problems/bacterial-wilt-cucurbits"),
    "S43": ("Univ. Wisconsin PDDC, Root rots in the garden", "https://pddc.wisc.edu/?p=4532"),
    "S44": ("UF/IFAS EDIS CG040, Citrus canker", "https://edis.ifas.ufl.edu/publication/CG040"),
    "S45": ("Univ. Minnesota Extension, Leaf mold of tomato", "https://extension.umn.edu/diseases/leaf-mold-tomato"),
    "S46": ("UC IPM, Pest Notes: Sooty mold (pn74108)", "https://ipm.ucanr.edu/PMG/PESTNOTES/pn74108.html"),
    "S47": ("Crop Protection Network, Gray leaf spot of corn", "https://cropprotectionnetwork.org/encyclopedia/gray-leaf-spot-of-corn"),
    "S48": ("UC IPM, Rice blast", "https://ipm.ucanr.edu/agriculture/rice/rice-blast/"),
    "S49": ("Crop Protection Network, Northern corn leaf blight", "https://cropprotectionnetwork.org/encyclopedia/northern-corn-leaf-blight-of-corn"),
    "S50": ("Shepherd et al. (2010), Maize streak virus, Mol. Plant Pathol. 11(1):1-12", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6640477"),
    "S51": ("Queensland Government, Black sigatoka of banana", "https://www.business.qld.gov.au/industries/farms-fishing-forestry/agriculture/biosecurity/plants/priority-pest-disease/black-sigatoka"),
    "S52": ("APS Plant Health Instructor, Black Sigatoka", "https://apsnet.org/edcenter/disandpath/fungalasco/pdlessons/Pages/BlackSigatoka.aspx"),
    "S53": ("UC IPM, Pest Notes: Aphids (pn7404)", "https://www.ipm.ucanr.edu/PMG/PESTNOTES/pn7404.html"),
    "S54": ("Texas A&M AgriLife, Iron deficiency symptoms in field crops", "https://agrilife.org/texasrowcrops/2023/12/11/iron-deficiency-symptoms-in-texas-field-crops/"),
    "S55": ("UC IPM, Phosphorus deficiency", "https://ipm.ucanr.edu/PMG/GARDEN/PLANTS/DISORDERS/phosphorusdeficiency.html"),
    "S56": ("UC IPM, Rusts", "https://ipm.ucanr.edu/home-and-landscape/rusts/"),
    "S57": ("Virginia Cooperative Extension, Master Gardener Handbook ch. 4: Plant pathology", "https://pressbooks.lib.vt.edu/emgtraining/chapter/4/"),
}

# key: (status, source ids, note)
M = {
    "healthy": ("supported", "", "no disease claim; reference description of a symptom-free leaf"),
    "rust": ("supported", "S56", ""),
    "common_rust": ("supported", "S1", ""),
    "brown_rust": ("supported", "S2", "wheat leaf (brown) rust"),
    "yellow_rust": ("supported", "S2;S3", ""),
    "stripe_rust": ("supported", "S3", ""),
    "bean_rust": ("supported", "S4", ""),
    "cedar_apple_rust": ("supported", "S5", ""),
    "red_rust": ("supported", "S6", "algal red rust (Cephaleuros virescens)"),
    "powdery_mildew": ("supported", "S7", ""),
    "mildew": ("supported", "S7", ""),
    "downy_mildew": ("supported", "S8", ""),
    "anthracnose": ("partial", "S9", "dark sunken lesions with darker borders supported; tip and edge dieback is host-dependent"),
    "brown_spot": ("supported", "S10", "rice brown spot"),
    "spot": ("supported", "S11", "generic leaf spot"),
    "bacterial_spot": ("supported", "S12", ""),
    "bacterial_leaf_spot": ("supported", "S12", ""),
    "black_rot": ("partial", "S13", "brown lesions with dark margins supported; yellowing not stated"),
    "black_spot": ("supported", "S14", ""),
    "leaf_blight": ("partial", "S11", "blights = large irregular dead areas; spread from edges/tips not stated"),
    "blight_leaf": ("partial", "S11", "as leaf_blight"),
    "alternaria_leaf_spot": ("supported", "S15", ""),
    "alternaria_leaf_blight": ("supported", "S16", ""),
    "alternaria_blotch": ("supported", "S17", ""),
    "early_blight": ("supported", "S18", ""),
    "late_blight": ("supported", "S19", ""),
    "target_spot": ("supported", "S20", ""),
    "mosaic_virus": ("supported", "S21", ""),
    "mosaic": ("supported", "S21", ""),
    "yellow_mosaic": ("supported", "S22", ""),
    "yellow_mosaic_virus": ("supported", "S22", "source is for mungbean yellow mosaic; the DPD class is on zucchini"),
    "curl": ("supported", "S23", "peach leaf curl"),
    "curl_virus": ("supported", "S24", ""),
    "yellow_leaf_curl_virus": ("supported", "S24", ""),
    "scab": ("supported", "S25", ""),
    "monillia_laxa": ("supported", "S26", ""),
    "brown_rot": ("supported", "S26", ""),
    "cercospora": ("supported", "S27", ""),
    "cercospora_leaf_blight": ("partial", "S28", "source says purple to bronze, leathery; 'dark brown spreading' not stated"),
    "frogeye_leaf_spot": ("supported", "S29", ""),
    "frog_eye_leaf_spot": ("supported", "S29", ""),
    "smut": ("supported", "S30", ""),
    "loose_smut": ("supported", "S30", ""),
    "bacterial_blight": ("partial", "S31", "angular water-soaked vein-bounded lesions supported; yellow margins not stated"),
    "angular_leaf_spot": ("supported", "S32", ""),
    "bacterial_leaf_streak": ("supported", "S33", ""),
    "scorch": ("partial", "S34", "marginal leaf scorch is one cultivar-dependent symptom of blueberry scorch; strawberry leaf scorch shows purple blotches instead"),
    "septoria_leaf_spot": ("supported", "S35", ""),
    "septoria": ("supported", "S35", ""),
    "ring_spot": ("partial", "S36", "papaya ringspot: rings on fruit, mosaic on leaves"),
    "sheath_blight": ("supported", "S37", ""),
    "spider_mites": ("supported", "S38", ""),
    "gray_mold": ("supported", "S39", ""),
    "phytophthora": ("supported", "S40", ""),
    "fusarium_wilt": ("supported", "S41", ""),
    "bacterial_wilt": ("supported", "S42", ""),
    "root_rot": ("supported", "S43", ""),
    "canker": ("supported", "S44", "citrus canker"),
    "leaf_mold": ("supported", "S45", ""),
    "sooty_mould": ("supported", "S46", ""),
    "gray_leaf_spot": ("supported", "S47", ""),
    "blast": ("supported", "S48", ""),
    "leafblast": ("supported", "S48", ""),
    "northern_leaf_blight": ("supported", "S49", ""),
    "streak_virus": ("supported", "S50", ""),
    "sigatoka": ("supported", "S51", ""),
    "segatoka": ("supported", "S51", ""),
    "black_leaf_streak": ("supported", "S52", ""),
    "aphis_spp": ("supported", "S53", ""),
    "nutrition_deficiency": ("partial", "S54;S55", "interveinal yellowing (Fe) and purpling (P) supported separately"),
    "white_spot": ("none", "", "no credible source found for tea white spot as small white spots; tea white scab literature differs"),
    "yellowing": ("supported", "S57", "general chlorosis"),
}

if __name__ == "__main__":
    sym = json.loads((HERE / "symptoms.json").read_text(encoding="utf-8"))
    assert set(sym) == set(M), set(sym) ^ set(M)
    with open(HERE / "symptoms_sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["disease", "description", "status", "sources", "note", "urls"])
        for k, d in sym.items():
            st, ids, note = M[k]
            w.writerow([k, d, st, ids, note, " ".join(S[i][1] for i in ids.split(";") if i)])
    from collections import Counter
    print(Counter(v[0] for v in M.values()), len(S), "sources")
