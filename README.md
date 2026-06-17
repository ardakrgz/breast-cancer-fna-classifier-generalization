# Cross-Cohort Generalization & Concept Transfer: WDBC vs WBCD

Final project for a Masters-level machine learning course.

Two-part study using the two classic Wisconsin breast cancer datasets from Dr. Wolberg's FNA lab (UW Madison). Both target the same binary task (benign/malignant) via the same clinical procedure (Fine Needle Aspirate), but were collected 4 years apart with entirely different feature encodings:

| Dataset | Year | Samples | Features |
|---------|------|---------|----------|
| [WDBC (UCI #17)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 1995 | 569 | 30 continuous image-derived measurements |
| [WBCD (UCI #15)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) | 1991 | 699 | 9 integer clinician-scored cytological features |

---

## Part 1 — Cross-Cohort Generalization

**[`wdbc_vs_wbcd_generalization.md`](wdbc_vs_wbcd_generalization.md)**

**Question:** Do the classifier rankings reported in Chaurasia & Pal (2020) on WDBC hold when the same models are trained independently on the older WBCD cohort?

Six classifiers (CART, SVM, NB, KNN, LR, MLP) are evaluated with 10-fold stratified CV on both datasets. Rank stability is measured with Spearman correlation.

**Result:** Spearman ρ = 0.829 (p = 0.042). Classifier ranking is largely stable across the two independent cohorts. CART, NB, and KNN hold their exact rank on both datasets. The only instability is among SVM, LR, and MLP at the top three positions, where AUC differences are within noise.

![Rank stability across cohorts](output_24_0.png)

---

## Part 2 — Concept-Level Transfer

**[`wdbc_wbcd_concept_transfer.md`](wdbc_wbcd_concept_transfer.md)**

**Question:** Can a model trained on one dataset classify cases from the other, using a hand-engineered semantic bridge between the two incompatible feature spaces?

A 4-concept bridge maps both feature vocabularies into a shared representation:

| Concept | WDBC features | WBCD features |
|---------|--------------|---------------|
| Size | radius, area, perimeter (mean / SE / worst) | clump thickness, single epithelial size |
| Shape irregularity | concavity, concave points (mean / SE / worst) | cell shape uniformity, marginal adhesion |
| Texture variance | texture (mean / SE / worst) | cell size uniformity, normal nucleoli |
| Compactness / chromatin | compactness, smoothness (mean / SE / worst) | bare nuclei, bland chromatin |

Within-dataset CV on just these 4 averaged features nearly matches full-feature performance, confirming the concepts are genuinely informative. Models are then trained on one dataset and evaluated on the other — zero-shot cross-dataset transfer.

**Result:** Transfer AUC reaches 0.972 (WBCD→WDBC) and 0.992 (WDBC→WBCD) for LR. A permutation test (200 shuffles of concept assignments) confirms the bridge structure is the driver — random assignments drop to ~0.55 AUC (p < 0.001 both directions).

![Within-dataset vs transfer AUC](output_12_0.png)

---

## Limitations

- Both datasets originate from the same lab, which is the weakest possible test of generalization. The high transfer AUC partly reflects the fact that both tasks are inherently easy (within-dataset AUC already > 0.99 for most classifiers).
- The concept bridge was constructed with knowledge of both feature sets and is not an independent prior. Future work could derive groupings from clinical ontologies or a third cohort.
- Models are trained on the full source dataset with no held-out source split, so within-domain generalization error is uncontrolled — most consequential for CART and KNN.

---

## Setup

```bash
pip install ucimlrepo scikit-learn pandas numpy matplotlib seaborn scipy
```

Both datasets are fetched automatically via `ucimlrepo` (UCI IDs 15 and 17) with a direct URL fallback. No manual download required.
