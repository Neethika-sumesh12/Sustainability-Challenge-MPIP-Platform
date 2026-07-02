# Multi-Modal Polymer Intelligence Platform (MPIP)
### A Technical Analysis for IBM Sustainable Innovation Prize 2026

---

## Glossary of Key Terms

Before diving into the analysis, here are all technical terms used throughout this document explained in plain English.

| Term | Full Form | What It Means |
|------|-----------|---------------|
| **NIR** | Near-Infrared Spectroscopy | A sensor that shines near-infrared light (wavelengths between 750–2500 nanometres) on a plastic item and reads the reflected light pattern to identify the material. The dominant plastic sorting technology used in recycling plants since the 1990s. |
| **MIR** | Mid-Infrared Spectroscopy | Uses a longer infrared wavelength range (2500–25000 nm). Unlike NIR, it detects fundamental molecular vibrations — giving a sharper chemical fingerprint. Critically, it works on black and dark plastics where NIR completely fails. |
| **XRF** | X-Ray Fluorescence | Fires X-rays at a material and measures the energy emitted to detect which chemical elements are present — for example, bromine in flame retardants, chlorine in PVC plastics, or lead and mercury in e-waste. Cannot identify the polymer type, but can detect hazardous chemical additives that NIR cannot. |
| **HSI** | Hyperspectral Imaging | A camera that captures hundreds of wavelength bands simultaneously for every pixel in an image — like a regular camera, but with a full chemical spectrum reading per pixel. Combines shape information with chemistry data. |
| **SWIR** | Short-Wave Infrared | A specific wavelength range (1000–2500 nm) used in hyperspectral cameras. Standard NIR and SWIR are the basis of most current industrial sorting cameras. |
| **FTIR** | Fourier-Transform Infrared Spectroscopy | A high-precision infrared analysis technique that measures a full spectrum at once using a mathematical process called Fourier Transform. More accurate than standard NIR but traditionally slower and used in labs rather than on conveyor belts. |
| **Raman Spectroscopy** | Raman Spectroscopy | Uses laser light to detect molecular vibrations. Complements NIR because it is sensitive to different chemical bonds — particularly useful for identifying polymer blends and additives. |
| **LIBS** | Laser-Induced Breakdown Spectroscopy | Fires a high-powered laser pulse to vaporise a tiny surface spot and analyses the emitted light. Detects elements including carbon, hydrogen, and metals — useful for composite and multilayer materials. |
| **RSC 2024 Paper** | Royal Society of Chemistry — Digital Discovery Journal, 2024 | The paper titled *"Sorting of polyolefins with near-infrared spectroscopy: identification of optimal data analysis pipelines and machine learning classifiers"*. Screened over 12,000 machine learning pipeline combinations to find the best NIR-only sorting approach. The most recent and comprehensive NIR-only study — and the starting point for understanding what NIR alone can and cannot do. |
| **MRF** | Materials Recovery Facility | An industrial sorting plant where mixed collected waste is separated into recyclable streams (plastic, paper, metal, glass). This is where plastic sorting machines operate at high conveyor belt speed — the real-world environment our solution must perform in. |
| **PCR** | Post-Consumer Recycled material | Plastic collected from consumers, sorted, processed, and converted into recycled pellets or flakes. Higher-purity PCR commands a higher market price. Contaminated or poorly-sorted PCR is low-value or unsellable. |
| **DPP** | Digital Product Passport | An EU-mandated electronic record (enforcement from 2027) that travels with a product through its lifecycle, containing data on materials, carbon footprint, and recyclability. All plastic packaging sold in the EU will require one. |
| **HDPE / LDPE / PP / PET** | High-Density Polyethylene / Low-Density Polyethylene / Polypropylene / Polyethylene Terephthalate | The most common plastic types. HDPE: rigid bottles, pipes. LDPE: soft films, shopping bags. PP: food containers, bottle caps. PET: drink bottles, food trays. HDPE, LDPE, and PP together form the polyolefin family — the hardest to distinguish by NIR. |
| **Carbon Black** | Carbon Black Pigment | A black pigment used in approximately 15–20% of all plastic products — electronics casings, automotive parts, black bin bags. Absorbs almost all NIR light, making these items invisible to standard sorting sensors. |
| **MMPSD** | Multi-modal Plastic Spectral Database | A research database created by Warwick University (2023) containing FTIR, Raman, and LIBS spectra for the same plastic samples — enabling multi-modal deep learning models to be trained across sensor types. |
| **SCAE** | Spectral Conversion Autoencoder | A deep learning model from the Warwick University paper that learns to convert spectral data from one sensor type into another — so a model trained on FTIR data can still work when only Raman data is available. |
| **IBM EIS** | IBM Environmental Intelligence Suite | An IBM software product that ingests environmental and sustainability data (satellite imagery, sensor feeds, operational data) for analysis and reporting. A natural platform layer for delivering MPIP's data outputs. |

---

## Section 1 — The Scale of the Problem

Plastic is one of the most produced materials on earth — and one of the least recovered.

- **9%** of all plastic produced globally is recycled. This figure has been essentially stagnant since 2019.
- **400 million tonnes** of plastic are produced every year, projected to double by 2050.
- **$57 billion** worth of recoverable material value is discarded annually as unrecycled plastic.
- **91%** of all plastic ends up in landfill, incineration, or the environment.

The instinctive assumption is that this is a collection problem — people are not separating their waste. But collection infrastructure has improved significantly over the past two decades without moving the recycling rate. The real bottleneck is **sorting accuracy**. Near-Infrared Spectroscopy (NIR) has been the dominant sorting technology since the 1990s and is deployed in virtually every recycling plant globally. Yet the recycling rate has not materially improved, because NIR has six critical failure modes that cause large volumes of recyclable plastic to be misidentified and sent to landfill.

---

## Section 2 — NIR Spectroscopy: Six Critical Failure Modes

NIR works by firing a beam of near-infrared light at a plastic item moving along a conveyor belt. The reflected light pattern — the spectrum — is compared against a database of known polymer spectra to identify the plastic type. In controlled laboratory conditions, this works well. In real-world recycling plants, it breaks down in the following six ways.

---

### Failure Mode 1 — Black and Dark Plastics: Complete NIR Blindness
**Severity: Critical**

Black plastics contain a pigment called carbon black, which absorbs almost all NIR wavelengths instead of reflecting them. The sensor receives essentially no meaningful signal — the item is invisible.

The Association of Plastic Recyclers (APR) officially states in its October 2024 updated report that NIR is **not a reliable technology** for sorting black plastics. Black and dark items account for 15–20% of all plastic by volume — including electronics casings, automotive components, and black refuse sacks. In current recycling plants, approximately 55% of detected black plastic by mass ends up in the residue line, sent to landfill or incineration regardless of what polymer it is made from.

The RSC 2024 paper itself found that **4 out of 5 HDPE misclassifications came from black-coloured HDPE samples** — confirming that colour alone is enough to defeat NIR identification.

---

### Failure Mode 2 — Polyolefin Sub-Class Overlap: A Fundamental Scientific Limit
**Severity: Critical**

Polyolefins — the polymer family including HDPE, LDPE, LLDPE (Linear Low-Density Polyethylene), and PP — are chemically almost identical. Their NIR spectra overlap severely. No matter how sophisticated the machine learning classifier placed on top, it is working with fundamentally similar input signals and will always produce systematic misclassifications.

Polyolefins are the **largest single polymer category by volume** in the plastic waste stream. Confusing HDPE with LDPE sends a high-value rigid plastic into a flexible film stream (or vice versa), destroying the value of both batches. The RSC 2024 paper screened over **12,000 machine learning pipeline combinations** specifically trying to resolve this — and still found systematic misclassifications between LDPE sub-types. The paper found **6 out of 7 LDPE misclassifications** came from a single polymer variant, confirming this is a structural spectral overlap problem, not a classifier design problem that more sophisticated algorithms can solve.

---

### Failure Mode 3 — Surface Contamination and Moisture Distort the Signal
**Severity: High**

Food residue, grease, moisture, soil, and adhesive labels on plastic surfaces alter NIR absorption bands — particularly at wavelengths around 1099–1172 nm and 1460–1677 nm. The sensor reads a contaminated spectrum rather than a clean polymer signature and misidentifies or rejects the item.

Real post-consumer plastic waste is almost always contaminated to some degree. A yoghurt pot with dairy residue, a shampoo bottle with traces of liquid, or a soiled carrier bag all produce distorted spectra. Yet every NIR reference database is built from **clean, laboratory-prepared samples**. A 2024 MDPI study found that biowaste contamination most severely distorted NIR spectra, with moisture creating additional absorption bands that can completely overwhelm the polymer-specific signal.

This is the primary reason why NIR achieves 95%+ accuracy in lab conditions (as shown in the RSC 2024 paper) but typically only 75–85% in real-world recycling plant conditions.

---

### Failure Mode 4 — Multilayer and Composite Packaging: Only the Surface Is Read
**Severity: High**

Modern food packaging increasingly uses multilayer films combining different materials — for example, PET + PE + Ethylene Vinyl Alcohol (EVOH) for barrier properties, or PE + aluminium foil + PP for shelf-stable pouches. NIR penetrates only a few micrometres into the surface. It reads either the top layer alone, or an interference signal combining layers, which matches no single polymer in the reference library.

Multilayer packaging is a growing share of consumer food packaging — snack bags, coffee pods, vacuum-sealed meat trays, cosmetic pouches, and most flexible stand-up pouches. These items are almost entirely **rejected by NIR sorting systems and sent to landfill**. All the material value — often multiple recoverable polymer types plus aluminium — is lost.

---

### Failure Mode 5 — Additives and Flame Retardants Distort the Polymer Fingerprint
**Severity: High**

Polymers contain many additives beyond their base chemistry: flame retardants (particularly bromine-based compounds in electronics plastics), plasticisers (in flexible PVC), UV stabilisers, colourants, and mineral fillers. These additives have their own NIR absorption signatures that overlap with or mask the base polymer signal.

Bromine-based flame retardants are especially common in e-waste — printer housings, television casings, computer enclosures. A NIR sensor cannot distinguish between a flame-retardant-containing ABS (Acrylonitrile Butadiene Styrene) casing and a clean ABS item. If these are sorted together, flame-retardant contamination renders the entire batch hazardous and unsellable.

The RSC 2024 paper explicitly acknowledged this as an open problem, stating that "a larger study with more samples and systematic representation of common additives may provide refinement." XRF (X-Ray Fluorescence) can detect bromine and other element-level markers of additives — but no published industrial system yet combines XRF intelligence with NIR polymer classification.

---

### Failure Mode 6 — Polymer Degradation Shifts Spectra Away from Reference Databases
**Severity: Medium**

UV exposure, heat cycling, and mechanical stress chemically alter polymers over time — breaking polymer chains, forming new functional groups, and shifting their infrared absorption spectra. A UV-weathered HDPE bottle produces a different NIR spectrum than a virgin HDPE sample in the reference database.

All NIR reference databases are built from virgin or near-pristine plastics. The RSC 2024 paper found that recycled HDPE and LDPE samples were misclassified specifically because of degradation artifacts from prior use. A separate study found prediction accuracy dropped by approximately 30% for PE and PP under aged conditions versus virgin samples. The older and more worn the plastic, the more likely it is to be misidentified and lost.

---

## Section 3 — What Already Exists: An Honest Survey of Prior Art

Before claiming anything as novel, it is essential to map what has already been researched and published. The following is an honest assessment.

### What Has Already Been Done

**RSC 2024 Paper (Sutliff et al., Royal Society of Chemistry — Digital Discovery Journal)**
The most comprehensive NIR-only optimisation study to date. Screened 12,000+ machine learning pipelines. Achieved excellent lab accuracy. Left open: black plastics, additives, contamination, and degradation. This paper defines the ceiling of what NIR alone can achieve.

**Multi-modal Spectral Database + Spectral Conversion Autoencoder — Warwick University (Neo et al., Journal of Cleaner Production, 2023)**
Combined FTIR, Raman, and LIBS data from the same plastic samples into a unified database (MMPSD). Introduced SCAE models that can translate between sensor types. Improved accuracy from 93.3% (single-sensor) to 97.0% (multi-sensor). However: lab-clean samples only, no real-time conveyor speed, no post-consumer contamination, no XRF, no industrial deployment.

**Bidirectional Cross-Attention Fusion of RGB + Hyperspectral NIR/SWIR (BCAF Paper, arxiv, 2026)**
Fused high-resolution RGB camera data with low-resolution hyperspectral NIR/SWIR using a bidirectional cross-attention Swin Transformer architecture. Achieved 76.4% mean accuracy at 31 images/second — real-time conveyor speed. However: NIR/SWIR wavelengths only (no Mid-Infrared, no XRF), black plastic still fails, no additive detection, no degradation scoring.

**Cross-Modal Swin Transformer for Plastic Identification (Ji et al., Waste Management, 2025)**
Multi-modal feature selection with a cross-modal transformer architecture. Confirmed in academic benchmarks that multimodal consistently outperforms single-sensor. However: academic benchmark only, no real MRF deployment, no industrial validation.

**Industrial Patents — NIR + MIR + XRF Hardware (AMP Robotics, ZenRobotics; US Patents 2021–2025)**
Hardware-level patents combining NIR, Mid-Wavelength Infrared, and XRF sensors on sorting lines targeting dark plastics. Tomra and Specim already ship commercial Mid-Infrared-enabled equipment for black plastic detection. However: rule-based fusion only — no learned, adaptive sensor weighting per item. No contamination intelligence. No degradation scoring. No material passport output.

**EU Digital Product Passport Regulation — Pilots (Digimarc, Circularise, SMX, 2024–2026)**
The EU has mandated Digital Product Passports for packaging from 2027. Multiple companies are piloting DPP systems using QR codes, RFID tags, and molecular markers. However: **all existing DPP systems start from a product with a known, embedded identifier**. None can generate a DPP from spectral analysis of an unlabelled, unknown post-consumer plastic item — which is the situation for 91% of what arrives at a recycling plant.

---

### Gap Map: What Remains Genuinely Unsolved

| Capability | Closest Existing Work | Status |
|---|---|---|
| NIR-only polyolefin classification | RSC 2024; multiple prior works | ✅ Solved in lab |
| Multi-spectral fusion (NIR + FTIR + Raman + LIBS) | Warwick MMPSD + SCAE (2023) | ✅ Solved in lab |
| RGB + Hyperspectral real-time fusion at conveyor speed | BCAF paper (2026) | ✅ Solved in lab |
| NIR + MIR + XRF hardware sensor combination | Tomra/Specim commercial; AMP/ZenRobotics patents | ⚠️ Hardware exists — AI layer missing |
| Black plastic detection via Mid-Infrared | Tomra commercial pilots (2024) | ⚠️ Pilot stage only — not scaled globally |
| Contamination-robust classification on real post-consumer waste | MDPI 2024 studied the effect — did not solve it | 🔴 Open gap |
| Degradation-aware polymer quality scoring at MRF speed | Lab degradation studies only | 🔴 Open gap |
| Dynamically learned sensor confidence weighting per item | Rule-based only in all patents and papers | 🔴 Open gap |
| AI-generated DPP from spectral data of unlabelled waste | All DPP systems require QR/RFID/label | 🔴 Open gap |
| Self-improving model from downstream QC failure feedback | Synthetic data augmentation explored — no closed loop | 🔴 Open gap |
| Material quality grade output beyond polymer class label | Class label only in all published work | 🔴 Open gap |

**The honest conclusion:** combining multiple sensors for plastic sorting is not new. What does not exist anywhere — in any published paper, patent, or commercial product — is a system that combines all four open gaps into a unified platform: contamination-aware adaptive sensor weighting, degradation scoring, spectral-to-DPP generation for unlabelled items, and a self-improving QC feedback loop.

---

## Section 4 — Proposed Solution: Multi-Modal Polymer Intelligence Platform (MPIP)

### Core Concept

MPIP does not simply combine sensors — existing hardware already does that at a basic level. The central innovation is the **AI reasoning layer** that sits above the sensor array: a system trained to dynamically decide which sensor to trust for each individual item, based on that item's specific surface condition, colour, and estimated composition.

The output is not just a class label (e.g., "this is HDPE"). It is a **rich material intelligence record**: polymer type, sub-grade, detected additives, surface contamination level, degradation score, and a full Digital Product Passport — generated entirely from spectral and visual data, without any QR code or embedded marker on the item.

---

### How MPIP Solves Each NIR Failure Mode

**Failure Mode 1 — Black plastics / NIR blindness**
Mid-Infrared Spectroscopy (MIR) operates at wavelengths that carbon black pigment does not absorb. MIR reads the fundamental molecular vibrations of the polymer directly, making the material's identity visible regardless of its colour. When the RGB (regular colour) camera detects a dark or black item, the IBM Granite AI model automatically increases the weight given to the MIR sensor and reduces reliance on the NIR channel for that specific item.

**Failure Mode 2 — Polyolefin sub-class overlap (HDPE vs LDPE vs PP)**
MIR's fingerprint region provides sharper molecular distinction between HDPE, LDPE, LLDPE, and PP than NIR's overtone bands. Additionally, the Granite model predicts physical properties — estimated density and crystallinity — from the combined multi-sensor signal, adding a third dimension of differentiation beyond spectral pattern matching alone.

**Failure Mode 3 — Surface contamination distorting spectra**
The RGB camera first assesses the item's surface state — detecting visible moisture, food residue, intact labels, or transparent areas. This surface assessment is passed to the Granite model as context. The model then dynamically down-weights the spectral bands most affected by contamination for that specific item, rather than applying a fixed, one-size-fits-all preprocessing step as all current systems do. A wet item and a greasy item trigger different band adjustments.

**Failure Mode 4 — Multilayer and composite packaging**
Short-Wave Infrared (SWIR) depth profiling can detect sub-surface layers in thin laminate films. XRF (X-Ray Fluorescence) detects metallic layers such as aluminium foil. The combined output generates a full material composition record for the item — listing all detected layers and their likely polymer types — rather than forcing a single incorrect class label onto a multi-material item and rejecting it.

**Failure Mode 5 — Additives and flame retardants**
XRF detects the elemental signature of halogen-based flame retardants (bromine, chlorine) and heavy metals (lead, mercury, cadmium) regardless of the base polymer. When XRF detects bromine alongside the NIR/MIR polymer signals, the Granite model flags the item as containing flame retardants (e-waste grade), preventing it from being classified as a clean, food-grade recyclable — and routing it to the correct specialist processing stream.

**Failure Mode 6 — Polymer degradation shifting spectra**
MPIP's reference model is trained on post-consumer waste samples across a range of degradation levels — not exclusively virgin plastics. Furthermore, the self-improving feedback loop (explained in Pillar 4 below) continuously updates the model with real-world outcome data from downstream quality checks. The system learns what degraded-but-sortable plastic looks like in practice. The output includes a degradation severity score (low / moderate / high / reject) to guide routing to the appropriate recycling pathway.

---

### The Four System Pillars

---

**Pillar 1 — Multi-Sensor Array: NIR + MIR + XRF + RGB**

A compact inline sensor unit mounted above the conveyor belt at the recycling plant. All four sensors fire simultaneously at every item passing beneath, at normal MRF conveyor speed (approximately 2–3 metres per second). Each sensor captures its data stream independently:

- **NIR (Near-Infrared):** Standard polymer identification for clean, non-black, non-degraded items. Handles the majority of sorting volume efficiently and cost-effectively.
- **MIR (Mid-Infrared):** Primary channel for black and dark-coloured plastics, polyolefin sub-classification, and fundamental molecular fingerprinting.
- **XRF (X-Ray Fluorescence):** Elemental analysis layer — detects flame retardants (bromine), heavy metals, chlorine (PVC), and aluminium (multilayer packaging). Triggered selectively based on signals from the other sensors.
- **RGB High-Resolution Camera:** Visual inspection layer — detects surface state (contamination level, moisture, label presence, colour, shape) and provides the context that the AI fusion model uses to calibrate its sensor weighting.

The hardware components individually already exist — Tomra sells Mid-Infrared-enabled sorting equipment, and XRF is standard in e-waste processing. What does not exist is the AI layer that intelligently combines them.

---

**Pillar 2 — IBM Granite Sensor Fusion Model: The Intelligence Layer**

The core innovation of MPIP. The Granite model does not simply concatenate all four sensor streams and apply a single classifier. It operates in two stages:

**Stage 1 — Surface State Assessment:** The RGB image is processed first to characterise the item's surface condition. This generates a confidence profile: which wavelength bands are likely to give reliable readings, and which are likely contaminated, masked by colour, or obscured by a label.

**Stage 2 — Adaptive Weighted Fusion:** Based on the surface state profile, the model dynamically adjusts how much weight it gives to each sensor channel for this specific item.
- A clean, white HDPE bottle: NIR weighted heavily, XRF minimal.
- A black electronics casing: MIR weighted heavily, NIR near-zero.
- A soiled multilayer pouch: XRF weighted for layer detection, MIR for base polymer, NIR down-weighted in contaminated spectral bands.

**Output:** Polymer type, sub-grade (e.g., HDPE food contact grade vs. HDPE industrial grade), detected additives (flame retardant flag, plasticiser flag), surface contamination level, degradation score, and recommended recycling pathway. This is a fundamentally richer output than any current sorting system produces.

---

**Pillar 3 — Material Intelligence API and Digital Product Passport Generator**

Every identified item generates a structured digital record transmitted via an open Application Programming Interface (API) in real time. For items with sufficient identification confidence, a full Digital Product Passport (DPP) is generated, compliant with the EU DPP regulation mandatory from 2027.

This is the key differentiator from all existing DPP systems. Every current DPP pilot — Digimarc, Circularise, SMX, and others — requires the product to have an **embedded identifier** inserted at the point of manufacture: a QR code, RFID tag, or molecular marker. These systems can only generate a passport for products that already have a known identity.

MPIP generates a DPP from spectral and visual analysis alone — for items with no label, no QR code, no marker. This covers the **91% of post-consumer plastic that arrives at a recycling plant with no digital identity whatsoever**.

The Material Intelligence API is an open, standardised interface. Downstream chemical recyclers, Post-Consumer Recycled (PCR) material buyers, brand-owner sustainability teams, and government regulators can all access verified material quality data through a single connection. This open-access layer creates network effects: the more facilities that connect, the richer and more comprehensive the material intelligence dataset becomes. This is the equivalent of **pfasID** in the 2025 IBM SIP winning solution (Safer Materials Advisor) — an open tool that created public value and compounding adoption beyond the core product.

Revenue model: SaaS subscription for recycling plant operators, per-passport fee for DPP generation, data analytics subscription for PCR buyers requiring verified material quality data.

---

**Pillar 4 — Self-Improving Feedback Loop from Downstream Quality Control Data**

The system learns from its own real-world mistakes — a capability no existing published sorting system has demonstrated. The feedback loop operates as follows:

1. **Sorting:** MPIP identifies and routes plastic items at the recycling plant. Each item's spectral data and AI-assigned classification are logged with a unique identifier.

2. **Downstream Quality Check:** The recycler or PCR buyer receiving the sorted material performs quality checks. Contamination incidents — for example, a batch of "HDPE" containing LDPE, or flame-retardant material mixed into a food-grade stream — are recorded.

3. **Failure Attribution:** Quality failure records are mapped back to the original items in the sorting log, identifying which spectral patterns were misclassified and under what conditions.

4. **Model Retraining:** These real-world failure cases are added to the training dataset with correct labels. The Granite model is retrained periodically — weekly or monthly — using the accumulated real-world data.

**Result:** The system continuously improves from real recycling plant conditions — contaminated, degraded, black, and composite plastics that are actually present in post-consumer waste streams. Each facility that deploys MPIP makes the model better for all other deployments. This is the same compounding improvement mechanism that made the 2025 SIP winning solution, Safer Materials Advisor, compelling to judges — it does not stand still after the initial launch.

---

### IBM Fit and Deployment Path

MPIP is built on existing IBM technology assets and creates clear new revenue streams across IBM's business units.

| IBM Asset | Role in MPIP |
|---|---|
| **IBM Granite Foundation Models** | Core AI engine for sensor fusion, adaptive weighting, degradation scoring, and DPP generation. |
| **IBM Environmental Intelligence Suite (EIS)** | Platform for ingesting sensor data streams, managing the Material Intelligence API, and delivering sustainability dashboards to enterprise clients. |
| **IBM Consulting** | Deployment engagements at recycling plant operator clients (Veolia, Suez, municipal operators). Integration with enterprise Scope 3 emissions reporting systems for Fortune 500 clients who need verified PCR material data. |
| **IBM Research** | Collaboration on the self-improving model architecture, particularly federated learning — enabling multiple recycling plants to share model intelligence without sharing raw operational data. |
| **IBM Cloud** | Hosting the Material Intelligence API, DPP generation service, and model retraining pipeline. |
| **Pilot candidate** | IBM's own facilities waste management operations as Client Zero — the same model used by the 2025 SIP winning Safer Materials Advisor, which piloted at IBM Yorktown and IBM Bromont before external deployment. |

---

## Section 5 — How This Satisfies the Prize Criteria

### Sustainability

*Does the solution create meaningful, long-term value supported by data-driven quantification? Does it scale and endure beyond a one-off win?*

MPIP directly attacks the structural bottleneck that has kept the global plastic recycling rate stagnant at 9% for years. Improving sorting accuracy at the recycling plant gate — the single point through which all recyclable plastic must pass — has a compounding, permanent effect on recycling outcomes.

**Quantified baseline:** If MPIP improves real-world sorting accuracy from a typical MRF rate of approximately 80% to 95% — a conservative estimate given the specific failure modes being addressed — the yield of high-grade recyclable material increases by 15 percentage points. At global production volumes of 400 million tonnes per year, this represents **60 million additional tonnes per year** diverted from landfill and incineration at scale. Even at a conservative pilot scale of 5 recycling plants processing 50,000 tonnes per year each, the pilot alone prevents 37,500 tonnes of misidentified plastic from going to landfill annually.

**Regulatory permanence:** The EU Digital Product Passport mandate (enforcement from 2027) makes MPIP's spectral-to-passport pipeline a regulatory compliance necessity, not a discretionary investment. This guarantees long-term demand and ensures the solution endures as permanent recycling infrastructure.

**Self-improving compounding value:** Unlike a one-off tool, MPIP becomes more accurate with each year of deployment as real-world QC feedback accumulates — delivering increasing value over time.

---

### Innovation

*How creative, unexpected, or bold is the idea? Did the team challenge assumptions and work beyond the boundaries of their day-to-day role?*

**The bold assumption being challenged:** The entire plastic sorting research field — including the RSC 2024 paper, and every academic and industrial effort before it — treats this as a spectroscopy optimisation problem. Every published paper asks: *"How can we squeeze better accuracy from our spectral data?"* MPIP reframes the question entirely: *"Which sensor should this system trust for this specific item, given that item's physical condition — and what richer information should the output contain beyond a simple class label?"*

This is precisely the kind of paradigm shift the prize rewards. The closest analogy among the 2025 prize winners is the **Maximo Visual Prompting Lab** (third place), which reframed an image annotation burden problem ("you need hundreds of training images") as a prompting problem ("show it one or two images"). MPIP makes the same type of reframe: from "better spectroscopy" to "smarter multi-sensor reasoning."

**Four specific novel contributions** — confirmed as not present in any published prior work as a combined deployed system:
1. Contamination-aware, dynamically learned sensor confidence weighting per individual item — not rule-based, not fixed preprocessing.
2. Polymer quality grading output (degradation score + additive flags) at recycling plant conveyor speed — no existing sorting system produces this.
3. Spectral-to-DPP pipeline for completely unlabelled post-consumer items with no embedded identifier — no current DPP system addresses this.
4. Self-improving closed loop from downstream QC failure data back into model retraining — no deployed sorting system has demonstrated this.

---

### Impact

*How widely does the solution resonate? Does it deliver tangible, concrete outcomes that others can build on or adopt?*

**IBM clients and Fortune 500 companies:** Every major consumer goods company — Unilever, Procter & Gamble, Nestlé, Coca-Cola — has committed to recycled content targets and Scope 3 greenhouse gas emissions reporting. They need verified, quality-graded Post-Consumer Recycled material data. MPIP's Material Intelligence API provides exactly that — a verified, DPP-compliant data feed for their sustainability reports, directly serviceable by IBM Consulting as a client engagement.

**Recycling plant operators:** Veolia, Suez, Biffa, Republic Services, and hundreds of municipal recycling facility operators globally face regulatory pressure to improve recycling rates. MPIP installs as an upgrade overlay to existing NIR conveyor infrastructure — no full equipment replacement required. The SaaS subscription model keeps upfront costs low for facility operators.

**Government and regulators:** The EU DPP regulation creates a legal mandate with a hard deadline. Municipal governments with recycling rate targets have a direct financial incentive to adopt systems that increase sorted material yield. The open Material Intelligence API allows governments to access aggregated regional recycling quality data for policy planning — without needing to build their own data infrastructure.

**Market scale:** The Material Passport Systems market for circular plastics is projected to reach $900 million in 2026, growing to $3.26 billion by 2036 at a compound annual growth rate of 13.7%. The EU's Carbon Border Adjustment Mechanism, which is expanding to include polymers, means that verified material passports with documented lower carbon content have direct financial value as they reduce import tariffs on recycled plastic. IBM is positioned to capture a meaningful and growing share of this market through MPIP's API and consulting services.

**Quantified pilot targets for a 5-facility pilot:**
- 37,500 additional tonnes of plastic recovered per year from misidentification errors alone.
- 100% of sorted items receive a Digital Product Passport — creating the first large-scale verified DPP dataset for post-consumer plastics.
- Estimated 15–20% improvement in PCR material market value, as higher quality grades command higher prices, generating direct revenue uplift for recycling plant operators.
- Accuracy improvement of approximately 3–5% per year as QC feedback data accumulates — a compounding return on investment.

---

## Section 6 — How to Frame This for the Submission

The strongest and most accurate way to position MPIP for the IBM Sustainable Innovation Prize is not as a "multimodal sensor fusion" project — that framing already exists in research papers and industrial patents and will be correctly identified as prior art by judges.

The correct positioning is this:

> *"An AI system that generates verifiable quality grades and EU-compliant Digital Product Passports for post-consumer plastics that currently have no identity — bridging the gap between the EU's 2027 DPP mandate and the 91% of plastic that will never carry a QR code."*

This framing is:
- **Genuinely novel** — no existing system does this
- **Regulation-anchored** — the EU 2027 mandate creates a hard deadline and guaranteed demand
- **Quantifiable** — 91% of post-consumer plastic, $57 billion in lost material value, 9% global recycling rate
- **Architecturally parallel to the 2025 winning submission** — Safer Materials Advisor used AI to generate open, verifiable chemical intelligence for a problem that was previously manual, slow, and opaque. MPIP does exactly the same for plastic material quality

The Material Intelligence API is MPIP's equivalent of pfasID — the open-access tool that gave SMA its network effect and public good dimension. IBM's own facilities as the pilot site mirrors SMA's IBM Yorktown and IBM Bromont deployment. The self-improving QC feedback loop mirrors SMA's expanding coverage from PFAS to heavy metals, flame retardants, and solvents.

The pattern fits. The gap is real. The evidence is quantified.

---

*Analysis prepared based on: RSC Digital Discovery 2024 paper (Sutliff et al.), Warwick University MMPSD + SCAE study (Neo et al., Journal of Cleaner Production, 2023), BCAF hyperspectral fusion paper (arxiv 2603.13941, 2026), Ji et al. Waste Management (2025), EU Digital Product Passport Regulation (adopted May 2024), APR NIR Technology Update (October 2024), MDPI contamination study (2024), PMC spectral augmentation study (2025), and IBM SIP 2025 winner documentation.*
