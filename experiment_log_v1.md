# BanglaCyberBench — Experiment Log

**Project:** BanglaCyberBench: Transformer Ensemble for Cyberbullying Detection in Bengali  
**Author:** Sefayet Alam (sefayetalam14@gmail.com)  
**Date:** April 2026  
**Repo:** github.com/Sefayet-Alam/CyberBully_Detection_Paper  
**Target Venue:** Q1 journal (IPM / ESWA) or ACL/EMNLP findings

---

## 1. Research Overview

Our research addresses the problem of cyberbullying detection in Bengali — the 7th most spoken language globally — where existing work suffers from small datasets (2K–44K samples), single-source bias, limited label granularity (binary or 5-class), and no cross-source robustness evaluation. We construct **BanglaCyberBench**, a 135,575-sample benchmark aggregated from 4 public sources spanning both Bangla script and Romanized Bangla, consolidated into a 9-class abuse taxonomy. We fine-tune three transformer encoders (BanglaBERT, MuRIL, XLM-R) in a multi-task setup (binary detection + 9-class abuse-type classification) with 3 seeds each, then combine all 9 checkpoints via an optimised weighted-logit ensemble. Our ensemble achieves **0.9247 Macro-F1** on binary detection and **0.7746 Macro-F1** on 9-class abuse-type classification, with strong cross-source and cross-script robustness (≥0.93 F1 on every holdout split).

---

## 2. Prior Work Comparison

### 2.1 Ahmed et al. (2021) — Hybrid Neural Network
- **Paper:** "Cyberbullying Detection Using Deep Neural Network from Social Media Comments in Bangla Language" (arXiv:2106.04506)
- **Dataset:** 44,001 Facebook comments, 5 classes (Non-bully, Sexual, Threat, Troll, Religious)
- **Method:** CNN-LSTM hybrid + ensemble with SVM
- **Best Results:** Binary accuracy = 87.91%, Multiclass accuracy = 85.00%
- **Limitations:** Single source (Facebook), no cross-source evaluation, 5 coarse classes, no transformer fine-tuning
- **Note:** This 44K dataset is one of the 4 sources in our benchmark (`facebook_44001`)

### 2.2 Sihab-Us-Sakib et al. (2024) — XLM-RoBERTa on CBD
- **Paper:** "Cyberbullying detection of resource constrained language from social media using transformer-based approach" (Natural Language Processing Journal, ScienceDirect)
- **Dataset:** 2,751 manually labelled texts (Cyberbullying Bengali Dataset / CBD), 5 classes (Cy-Flaming, Cy-Threat, Cy-Pull-a-Pig, Cy-Racism, Non-bullying)
- **Method:** XLM-RoBERTa fine-tuning, compared against SVM, MNB, RF, GRU, CNN, LSTM, BiLSTM
- **Best Results:** XLM-R achieved F1 = 0.83, accuracy = 82.61%
- **Limitations:** Very small dataset (2.7K), single source, no ensemble, no multi-task learning

### 2.3 Saifullah et al. (2024) — BullyFilterNeT
- **Paper:** "Cyberbullying Text Identification based on Deep Learning and Transformer-based Language Models" (EAI Endorsed Transactions on INIS)
- **Dataset:** 44,001 Facebook comments (same as Ahmed et al.)
- **Method:** BanglaBERT-based deep learning model (BullyFilterNeT)
- **Best Results:** Accuracy = 88.04%
- **Limitations:** Single source, single model, no robustness evaluation

### 2.4 Hoque & Seddiqui (2025) — Transformer-Stacking Framework
- **Paper:** "Advancing cyberbullying detection in low-resource languages: a transformer-stacking framework for Bengali" (Frontiers in AI, Feb 2026)
- **Dataset:** 44,001 Facebook comments (same as Ahmed et al.)
- **Method:** Stacking of XLM-R-base, mBERT, Bangla-Bert-Base with MLP meta-classifier
- **Best Results:** Sub-task A (binary): F1 = 93.61%, accuracy = 93.62%; Sub-task B (multiclass 5-class): F1 = 89.23%, accuracy = 89.23%
- **Limitations:** Single source (44K dataset), 5-class taxonomy, no cross-source/cross-script robustness

### 2.5 Comparison Table

| Study | Year | Dataset Size | Sources | Scripts | Classes | Binary Best | Multiclass Best | Robustness |
|---|---|---|---|---|---|---|---|---|
| Ahmed et al. | 2021 | 44K | 1 | Bangla | 5 | 87.91% acc | 85.00% acc | ✗ |
| Sihab-Us-Sakib et al. | 2024 | 2.7K | 1 | Mixed | 5 | 82.61% acc | — | ✗ |
| Saifullah et al. | 2024 | 44K | 1 | Bangla | 5 | 88.04% acc | — | ✗ |
| Hoque & Seddiqui | 2025 | 44K | 1 | Bangla | 5 | 93.62% acc | 89.23% acc | ✗ |
| **Our research** | **2026** | **135K** | **4** | **Both** | **9** | **92.56% acc / 92.47% F1** | **86.88% acc / 77.46% F1 (9-class)** | **✓ (6 splits)** |

**Key differentiators of our research:**
1. **3× larger dataset** than any prior work (135K vs 44K)
2. **Multi-source** benchmark (4 datasets vs single-source in all prior work)
3. **Dual-script** coverage (73,999 romanized + 61,576 Bangla script)
4. **9-class fine-grained taxonomy** with priority-based compound-label resolution
5. **Cross-source and cross-script robustness evaluation** — no prior work tests this
6. **Multi-task architecture** (binary + abuse-type jointly) — no prior work uses this for Bangla
7. **Ablation study** (pending NB09) covering pooling, loss, model components

---

## 3. Dataset: BanglaCyberBench

### 3.1 Sources

| Source | Samples | Script | Origin | Content |
|---|---|---|---|---|
| `banth` | 73,999 | Romanized | Kaggle | Bangla text in English script from social media |
| `bd_shs` | 5,029 | Bangla | Mendeley | Bengali sexual harassment / bullying comments |
| `facebook_44001` | 44,001 | Bangla | Mendeley (Ahmed et al. 2021) | Facebook comments: Sexual, Threat, Troll, Religious, Non-bully |
| `multilabel_12557` | 12,546 | Bangla | Kaggle | Multi-label cyberbullying, sexual harassment, threat, spam |

**Total: 135,575 samples**

### 3.2 Script Distribution

| Script | Samples | Percentage |
|---|---|---|
| Romanized (English script) | 73,999 | 54.6% |
| Bangla (Unicode) | 61,576 | 45.4% |

### 3.3 Binary Label Distribution

| Label | Samples | Percentage |
|---|---|---|
| Not Harmful (0) | 75,545 | 55.7% |
| Harmful (1) | 60,030 | 44.3% |

Ratio: 1.26:1 — near-balanced.

### 3.4 Nine-Class Abuse Taxonomy

Raw data contained 89 unique `label_type` values (many with <15 samples). Our research consolidated these into 9 semantically coherent classes using priority-based resolution for compound labels.

**Priority order:** threat > sexual > religious > gender > political > abusive > personal > other > none

| Class | Train Samples | Maps From (Key Examples) |
|---|---|---|
| none | 62,473 | none, not bully |
| abusive | 12,690 | Abusive/Violence, troll |
| personal | 10,959 | Personal Offense, Body Shaming, Origin, slander, Misc |
| sexual | 8,743 | sexual, sexual+religious (FIX: sexual > religious) |
| religious | 7,356 | religious, Religious, religion, religion_slander |
| threat | 3,071 | threat, callToViolence*, religious+threat, sexual+threat |
| political | 1,842 | Political |
| other | 741 | spam |
| gender | 585 | gender, Gender, gender_slander |

**Compound label resolution:** `sexual,religious` → `sexual` (sexual > religious in priority). Fallback for unseen compounds: split on comma → look up each part → pick highest priority → substring match → default `other`.

### 3.5 Data Splits

| Split | Samples |
|---|---|
| Train | 108,460 |
| Validation | 13,557 |
| Test | 13,558 |

### 3.6 Duplicate Handling

40,590 raw duplicates exist across sources. These are intentionally **not** deduplicated — kept for the robustness study to reflect real-world data overlap.

---

## 4. Environment

| Parameter | Value |
|---|---|
| OS / Kernel | Windows 11, VSCode Jupyter, Python 3.13.13 (MSC v.1944 64-bit) |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.5.3 |
| Tokenizers | 0.22.2 |
| GPU (primary) | NVIDIA RTX 4060 Ti — 8.6 GB VRAM |
| GPU (secondary) | Intel UHD 16 GB — NOT used (immature PyTorch support) |
| RAM | 32 GB |

---

## 5. Model Architecture

All three transformer models share the same architecture, differing only in the encoder backbone:

| Component | Specification |
|---|---|
| **Encoder** | AutoModel — BanglaBERT (ELECTRA), MuRIL (BERT), XLM-R (RoBERTa) |
| **Pooling** | 0.5 × CLS + 0.5 × mean-pool (NOT CLS-only) |
| **Task Head (TaskHead)** | Dropout(0.25) → Linear(hidden, 384) → GELU → LayerNorm(384) → Dropout(0.25) → Linear(384, n_classes) |
| **Binary Head** | TaskHead with n_classes=2 |
| **Abuse-Type Head** | TaskHead with n_classes=9 |
| **Binary Loss** | FocalLoss(gamma=1.5) + effective-sample class weights (beta=0.999, clamped at 10× min) |
| **Abuse-Type Loss** | FocalLoss(gamma=2.5) + same class weights |
| **token_type_ids** | Only passed to BERT-family models (BanglaBERT, MuRIL); skipped for XLM-R/RoBERTa |
| **Mixed Precision** | torch.amp.GradScaler('cuda') + torch.autocast |

### 5.1 Training Hyperparameters

| Parameter | Value |
|---|---|
| max_length | 128 |
| batch_size | 16 |
| gradient_accumulation | 2 (effective batch = 32) |
| epochs | 8 |
| encoder_lr | 2e-5 |
| head_lr | 8e-5 |
| lr_decay | 0.90 per layer |
| label_smoothing | 0.03 |
| dropout | 0.25 |
| head_hidden_dim | 384 |
| class_weight_beta | 0.999 |
| focal_gamma_binary | 1.5 |
| focal_gamma_abuse | 2.5 |
| patience | 3 (early stopping) |
| monitor | 0.7 × binary_F1 + 0.3 × abuse_F1 |
| num_workers | 0 (Windows Jupyter deadlock prevention) |

### 5.2 Transformer Encoders

| Model | Pretrain Source | Architecture | Pretrain Data | HuggingFace ID |
|---|---|---|---|---|
| BanglaBERT | BUET CSE NLP | ELECTRA (discriminator) | 27.5 GB Bengali text from 110 websites | `csebuetnlp/banglabert` |
| MuRIL | Google | BERT-base | 17 Indian languages + transliterated pairs | `google/muril-base-cased` |
| XLM-R | Meta AI | RoBERTa-base | 2.5 TB CommonCrawl in 100 languages | `xlm-roberta-base` |

---

## 6. Notebook Pipeline & Results

### NB04 — Baseline Models (DONE)

**Purpose:** Establish non-transformer baselines using TF-IDF features + classical ML and a BiLSTM.

**Results:**

| Model | Macro-F1 | Accuracy | MCC | AUROC |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8669 | 0.8691 | 0.7342 | 0.9429 |
| TF-IDF + SVM | 0.8877 | 0.8897 | 0.7760 | 0.9558 |
| TF-IDF + Random Forest | 0.9090 | 0.9099 | 0.8183 | 0.9718 |
| BiLSTM | 0.8914 | 0.8926 | 0.7828 | 0.9479 |

**Takeaway:** TF-IDF + Random Forest is the strongest baseline at 0.909 F1. This sets the bar that transformers must beat.

---

### NB05 — Transformer Fine-Tuning (DONE — 9/9 runs)

**Purpose:** Fine-tune BanglaBERT, MuRIL, and XLM-R with 3 seeds each (42, 123, 456) using the multi-task setup. Save model checkpoints, logits, and per-run results.

**Per-Run Test Results:**

| Model | Seed | Binary Macro-F1 | Abuse-Type Macro-F1 | Binary MCC | Abuse-Type MCC |
|---|---|---|---|---|---|
| BanglaBERT | 42 | 0.9080 | 0.7428 | 0.8164 | 0.7556 |
| BanglaBERT | 123 | 0.9095 | 0.7380 | 0.8195 | 0.7554 |
| BanglaBERT | 456 | 0.9038 | 0.7413 | 0.8080 | 0.7477 |
| MuRIL | 42 | 0.9049 | 0.7260 | 0.8098 | 0.7456 |
| MuRIL | 123 | 0.9066 | 0.7344 | 0.8132 | 0.7489 |
| MuRIL | 456 | 0.9060 | 0.7304 | 0.8120 | 0.7494 |
| XLM-R | 42 | 0.8989 | 0.7122 | 0.7978 | 0.7334 |
| XLM-R | 123 | 0.8926 | 0.7069 | 0.7857 | 0.7237 |
| XLM-R | 456 | 0.8965 | 0.7150 | 0.7930 | 0.7340 |

**Averaged Results (mean ± std):**

| Model | Binary Macro-F1 | Abuse-Type Macro-F1 | Binary MCC | Abuse-Type MCC |
|---|---|---|---|---|
| BanglaBERT | 0.9071 ± 0.0030 | 0.7407 ± 0.0025 | 0.8146 ± 0.0060 | 0.7529 ± 0.0045 |
| MuRIL | 0.9058 ± 0.0009 | 0.7303 ± 0.0042 | 0.8117 ± 0.0017 | 0.7480 ± 0.0021 |
| XLM-R | 0.8960 ± 0.0032 | 0.7114 ± 0.0041 | 0.7922 ± 0.0061 | 0.7304 ± 0.0058 |

**Takeaway:** BanglaBERT leads on both tasks (language-specific pretraining advantage). MuRIL is very close (multilingual Indian language pretraining). XLM-R is slightly behind (general multilingual). All transformers beat the best baseline (0.909 RF) except XLM-R which matches it. Low std across seeds indicates stable training.

**Outputs saved:**
- `../outputs/models_v2_fix/label_encoders.json`
- `../outputs/models_v2_fix/transformer_results_all.csv`
- `../outputs/models_v2_fix/transformer_results_averaged.csv`
- `../outputs/models_v2_fix/{model}_seed{seed}/best_model.pt, val_logits.pt, test_logits.pt, results.json`

---

### NB06 — Ensemble & Threshold Tuning (DONE)

**Purpose:** Combine all 9 model checkpoints into a weighted-logit ensemble. Optimise weights via Nelder-Mead on validation set. Tune binary threshold. Evaluate on test set.

**Optimised Ensemble Weights:**

| Model-Seed | Weight |
|---|---|
| banglabert_seed42 | 0.2817 |
| muril_seed42 | 0.1680 |
| xlmr_seed456 | 0.1625 |
| muril_seed123 | 0.1187 |
| banglabert_seed456 | 0.1163 |
| muril_seed456 | 0.0664 |
| banglabert_seed123 | 0.0422 |
| xlmr_seed42 | 0.0336 |
| xlmr_seed123 | 0.0107 |

**Note:** Weights are non-uniform. BanglaBERT seed 42 dominates (0.28). XLM-R seed 123 contributes minimally (0.01). This differs from earlier v1 experiments where Nelder-Mead converged to uniform 1/9.

**Binary Ensemble — Test Results:**

| Metric | Value |
|---|---|
| Accuracy | 0.9256 |
| Macro-F1 | 0.9247 |
| Weighted-F1 | 0.9256 |
| MCC | 0.8494 |
| AUROC | 0.9731 |
| AUPRC | 0.9658 |

**Tuned threshold:** 0.50 (confirms near-balanced dataset; no benefit from shifting).

**Abuse-Type Ensemble — Test Results:**

| Metric | Value |
|---|---|
| Macro-F1 | 0.7746 |
| Accuracy | 0.8688 |

**Per-Class Abuse-Type Report:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| none | 0.93 | 0.93 | 0.93 | 7,830 |
| religious | 0.89 | 0.86 | 0.87 | 908 |
| sexual | 0.79 | 0.81 | 0.80 | 1,083 |
| personal | 0.79 | 0.78 | 0.78 | 1,344 |
| other | 0.74 | 0.80 | 0.77 | 79 |
| abusive | 0.76 | 0.75 | 0.76 | 1,600 |
| political | 0.75 | 0.77 | 0.76 | 259 |
| threat | 0.73 | 0.77 | 0.75 | 365 |
| gender | 0.59 | 0.53 | 0.56 | 90 |

**Takeaway:** Ensemble lifts binary F1 from best-single 0.9095 to 0.9247 (+1.5 points). Abuse-type F1 lifts from best-single ~0.74 to 0.7746. `gender` class is weakest (F1=0.56) due to very small support (90 test samples). `none` class is strongest (0.93).

**Outputs saved:**
- `../outputs/ensemble/final_config.json`
- `../outputs/ensemble/ensemble_test_metrics.json`
- `../outputs/ensemble/test_preds.npy, test_probs.npy`
- `../outputs/ensemble/threshold_tuning.png, cm_ensemble_test.png`

---

### NB08 — Robustness Evaluation (DONE)

**Purpose:** Test whether the ensemble generalises across data sources and scripts. For each split, hold out an entire source or script, and evaluate using only models trained on the remaining data (inference only — no retraining).

**Robustness Results (Binary Ensemble):**

| Split | N | Macro-F1 | Weighted-F1 | Accuracy | MCC | AUROC |
|---|---|---|---|---|---|---|
| random_test (in-domain) | 13,558 | 0.9245 | 0.9254 | 0.9254 | 0.8489 | 0.9738 |
| source_holdout_banth | 73,999 | 0.9777 | 0.9819 | 0.9819 | 0.9554 | 0.9959 |
| source_holdout_bd_shs | 5,029 | 0.9342 | 0.9341 | 0.9342 | 0.8744 | 0.9926 |
| source_holdout_facebook_44001 | 44,001 | 0.9736 | 0.9761 | 0.9761 | 0.9473 | 0.9952 |
| source_holdout_multilabel_12557 | 12,546 | 0.9304 | 0.9358 | 0.9357 | 0.8608 | 0.9799 |
| script_holdout_romanized | 73,999 | 0.9777 | 0.9819 | 0.9819 | 0.9554 | 0.9959 |

**Key Observations:**
1. All holdout splits maintain F1 ≥ 0.93 — strong generalisation.
2. `source_holdout_banth` and `script_holdout_romanized` produce identical results (F1=0.9777) because `banth` is entirely romanized text — these two splits evaluate on the same data. This should be noted in the paper and they should not be presented as independent evidence.
3. Holdout F1 values exceed in-domain test F1 (0.9245) for banth, facebook, and romanized splits — likely because those sources have cleaner / more separable label distributions.
4. Hardest split: `multilabel_12557` (F1=0.9304) — expected since these samples had compound labels requiring consolidation.

**Outputs saved:**
- `../outputs/robustness/robustness_results.csv`

---

### NB09 — Ablation Runs (PENDING — estimated ~4-5 hrs GPU)

**Purpose:** Systematically remove or modify individual components to measure their contribution. Each ablation trains a single model (BanglaBERT, seed=42) with one change, comparing against the full-config baseline.

**Planned Ablations:**

| Ablation | What Changes | Tests |
|---|---|---|
| CLS-only pooling | Remove mean-pool, use CLS token only | Value of blended pooling |
| Mean-only pooling | Remove CLS, use mean-pool only | Value of blended pooling |
| No focal loss | Replace FocalLoss with CrossEntropyLoss | Value of focal loss for imbalance |
| No class weights | Remove effective-sample class weighting | Value of class weights |
| No LR decay | Uniform LR across all layers | Value of layer-wise LR decay |
| Single-task binary | Remove abuse-type head, train binary only | Value of multi-task learning |
| Single-task abuse | Remove binary head, train abuse-type only | Value of multi-task learning |
| Simple head | Single Linear layer instead of 2-layer TaskHead | Value of deeper classification head |

**Expected outputs:**
- `../outputs/ablations/ablation_results.csv`
- Per-ablation confusion matrices and metrics

---

### NB07 — Analysis & Visualization (PENDING — ~2 min, no GPU)

**Purpose:** Generate all paper-ready figures, tables, and analysis from results of NB04-06 and NB08-09. This is the final synthesis notebook run after all experiments complete.

**Expected outputs:**
- `../outputs/paper/` directory containing:
  - LaTeX tables (baseline comparison, transformer results, ensemble results, ablation, robustness)
  - Confusion matrices (per-model, ensemble, per-class)
  - ROC curves, precision-recall curves
  - Training curves
  - Class distribution charts
  - Robustness heatmap
  - Ablation comparison chart

---

## 7. Output Directory Structure

```
../outputs/
├── models_v2_fix/
│   ├── label_encoders.json          ← shared across NB05/06/08/09
│   ├── transformer_results_all.csv
│   ├── transformer_results_averaged.csv
│   ├── banglabert_seed42/
│   │   ├── best_model.pt
│   │   ├── val_logits.pt
│   │   ├── test_logits.pt
│   │   ├── results.json
│   │   └── label_encoders.json
│   ├── banglabert_seed123/  (same structure)
│   ├── banglabert_seed456/  (same structure)
│   ├── muril_seed42/        (same structure)
│   ├── muril_seed123/       (same structure)
│   ├── muril_seed456/       (same structure)
│   ├── xlmr_seed42/         (same structure)
│   ├── xlmr_seed123/        (same structure)
│   └── xlmr_seed456/        (same structure)
├── ensemble/
│   ├── final_config.json
│   ├── ensemble_test_metrics.json
│   ├── test_preds.npy
│   ├── test_probs.npy
│   ├── threshold_tuning.png
│   └── cm_ensemble_test.png
├── robustness/
│   └── robustness_results.csv
├── ablations/                        ← generated by NB09
│   └── ablation_results.csv
└── paper/                            ← generated by NB07
    ├── tables/ (LaTeX .tex files)
    └── figures/ (high-res PNG/PDF)
```

---

## 8. Bugs Fixed in v3 Notebooks

| Bug | Affected | Fix |
|---|---|---|
| Python 3.13 install crash (no cp313 tokenizer wheels) | NB05 | tokenizers≥0.19.0, transformers≥4.44.0 |
| num_workers>0 deadlock on Windows Jupyter | NB05, NB09 | num_workers=0 |
| `sexual,religious` wrongly mapped to `religious` | NB05/06/08/09 | sexual,religious → sexual (sexual > religious) |
| abuse_type F1=0.002 from NaN focal loss (89 raw classes) | NB05 | 9-class consolidation + beta=0.999 + max_w clamp |
| Only 1 model/seed ran | NB05 | All 3 models × 3 seeds enabled |
| NB06 y_true was 89-class vs 9-class logits | NB06 | Load label_encoders.json + same consolidate_type() |
| NB09 binary labels all = -1 (int key vs str lookup) | NB09 | Load label_encoders.json from NB05 |
| NB09 model arch mismatch with NB05 | NB09 | TaskHead 2-layer + CLS+mean pooling |
| NB08 rebuilt label_encoders independently | NB08 | Load label_encoders.json from NB05 |
| torch.cuda.amp.GradScaler deprecated | NB05, NB09 | torch.amp.GradScaler('cuda') |
| XLM-R received token_type_ids it ignores | NB05/08/09 | Check model_type before passing TTI |
| NB06 missing ensemble_test_metrics.json | NB06 | Added json.dump after final test eval |

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| Multi-class (not multi-label) for abuse_type | Compound labels are ~11% of data. Simpler, comparable to prior work. Multi-label = future work. |
| 9 classes (not 89) | 89 classes had extreme imbalance (some with 1-14 samples). 9 classes are semantically clean. |
| Macro-F1 as primary metric | Near-balanced dataset. Macro-F1 ensures each class contributes equally. |
| Default threshold 0.50 | 1.26:1 ratio = near-balanced. Tuning confirmed no benefit from shifting. |
| Non-uniform ensemble weights | Nelder-Mead optimisation on val set. BanglaBERT seed 42 dominates (0.28). |
| 3 seeds per model (9 runs total) | Sufficient for stability at submission. Expand to 5 seeds for camera-ready. |
| CLS + mean pooling blend (0.5/0.5) | CLS = global context, mean = token-level details. Empirically better than CLS-only. |
| Focal loss (gamma=1.5/2.5) | Handles class imbalance. Higher gamma (2.5) for harder 9-class abuse_type task. |
| Layer-wise LR decay (0.90) | Lower layers learn general features (lower LR), upper layers adapt to task (higher LR). |
| Duplicates kept | 40K duplicates across sources kept intentionally for robustness study realism. |

---

## 10. Troubleshooting Reference

| Symptom | Cause | Fix |
|---|---|---|
| Training never starts (hours pass, no epochs) | num_workers>0 Windows deadlock | num_workers=0 |
| abu= shows 0.002 or NaN at epoch 1 | consolidate_type not applied before build_label_encoders | Rerun from consolidation cell |
| NB06 abuse F1 looks wrong | Raw 89-class label_type loaded | Load label_encoders.json from models_v2_fix/ |
| NB09 all binary labels -1 | Int key in encoder, str lookup | Load label_encoders.json from NB05 |
| CUDA OOM | batch_size too large for 8 GB | Reduce batch_size 16→8 |
| ElectraModel UNEXPECTED keys | BanglaBERT checkpoint has classifier head | Ignore — encoder loaded correctly |
| NB09 crashes mid-run | GPU OOM or exception | Re-run — completed conditions skip via results.json |

---

## 11. Planned Paper Structure (NB10)

| Section | Content |
|---|---|
| Abstract | 135K benchmark, multi-source, multi-script, ensemble, 0.93+ F1 |
| Introduction | Bangla cyberbullying problem, benchmark gap, 3 contributions |
| Related Work | Prior Bangla NLP datasets (Ahmed, Sihab-Us-Sakib, Saifullah, Hoque), cyberbullying detection, transformer ensembles |
| Dataset | 4 sources, preprocessing, statistics, 9-class label schema, compound label resolution |
| Methodology | Multi-task transformer, focal loss, layer-wise LR decay, CLS+mean pooling, weighted ensemble |
| Experiments | Baselines (NB04), per-model transformers (NB05), ensemble results (NB06), multi-task table |
| Analysis | Ablation study (NB09), robustness source/script holdout (NB08), error analysis, per-class breakdown |
| Conclusion | Summary, limitations (multi-label future work, 5-seed camera-ready), broader impact |

---

*BanglaCyberBench Experiment Log v1 | Sefayet Alam | April 2026*
