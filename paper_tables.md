# Research Paper Tables: Osteoporosis Diagnosis & Model Validation

This document compiles the empirical evaluation metrics, comparative benchmarks, and data leakage analyses conducted across the machine learning and deep learning pipelines in the OsteoCare-AI repository.

---

## Table 1: DEXA Tabular Model Benchmark & Leakage Analysis

Evaluation of the ExtraTrees Classifier (`n_estimators=400`, `max_depth=20`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features='sqrt'`).

| Pipeline Configuration | Test Set Sample Size ($N$) | Accuracy | Precision | Recall (Sensitivity) | Specificity (TNR) | F1-Score | ROC-AUC | 5-Fold CV Accuracy | 5-Fold CV ROC-AUC | Methodological Integrity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Leaky Baseline** (`model_training.ipynb`) | 388 (balanced synthetic) | **88.92%** | **86.47%** | **92.27%** | **85.57%** | **89.28%** | **0.9462** | 86.63% ± 0.97% | 0.9393 ± 0.012 | **Flawed**: Synthetic oversampling (SMOTE-like; generating code not preserved in repository) applied to entire dataset *prior* to train/test split. |
| **Leakage-Free (All 20 Features)** | 308 (real patients) | **79.22%** | **79.07%** | **59.65%** | **90.72%** | **68.00%** | **0.8748** | 81.07% ± 2.16% | 0.8856 ± 1.51% | **Valid**: 80/20 split first; median imputation, standard scaling, and random oversampling with replacement strictly on training partition. |
| **Leakage-Free (Without T-Scores)** | 308 (real patients) | **70.45%** | **70.91%** | **34.21%** | **91.75%** | **46.15%** | **0.7177** | 72.74% ± 2.12% | 0.7419 ± 3.18% | **Valid**: Evaluates non-invasive clinical chemistry & anthropometrics alone. |

> **Key Finding on Data Leakage & Feature Subsets:**
> On the leaky baseline, apparent sensitivity was **32.6 points higher** (92.27% vs. 59.65%) and test accuracy was 9.7 points higher (88.92% vs. 79.22%), though this compares two fundamentally different evaluation sets (388 resampled instances containing synthetic rows versus 308 untouched real patients). Furthermore, in this dataset, routine markers alone gave low sensitivity (**34.21%**), demonstrating that clinical chemistry and anthropometrics alone had limited detection capability compared to direct densitometric features.

---

## Table 2: Confusion Matrix Summary Across All Evaluated Models

| Model & Evaluation Setting | True Negative (TN) | False Positive (FP) | False Negative (FN) | True Positive (TP) | Total Test $N$ | Positive Class Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DEXA ExtraTrees — Leaky Baseline** | 166 | 28 | 15 | 179 | 388 | 50.0% (Synthetic/Resampled) |
| **DEXA ExtraTrees — Leakage-Safe (20 Features)** | 176 | 18 | 46 | 68 | 308 | 37.0% (Real) |
| **DEXA ExtraTrees — Leakage-Safe (No T-Scores)** | 178 | 16 | 75 | 39 | 308 | 37.0% (Real) |
| **Clinical Gradient Boosting — Without DEXA** | 147 | 0 | 39 | 108 | 294 | 50.0% (Public Kaggle) |

---

## Table 3: Without-DEXA Clinical Questionnaire Model (Gradient Boosting)

Trained on `original2.csv` ($N = 1,958$ raw rows, 16 attributes from public Kaggle dataset; 10 selected clinical features: `Age`, `Smoking`, `Physical_Activity`, `Medications`, `Vitamin_D_Intake`, `Hormonal_Changes`, `Gender`, `Body_Weight`, `Prior_Fractures`, `Alcohol_Consumption`). Evaluated with tuned hyperparameters: `n_estimators=300`, `max_depth=2`, `learning_rate=0.05`, `subsample=0.8`, `min_samples_leaf=1`.

| Metric Level | Metric | Value | Interpretation |
| :--- | :--- | :---: | :--- |
| **Overall** | Test Accuracy | **86.73%** | 255 correct out of 294 test instances |
| **Overall** | ROC-AUC | **0.8624** | Area under ROC curve |
| **Osteoporosis (Class 1)** | Precision | **100.00%** | Zero false positives in test split ($108 / 108$) |
| **Osteoporosis (Class 1)** | Recall (Sensitivity) | **73.47%** | 108 caught out of 147 true positive records |
| **Osteoporosis (Class 1)** | F1-Score | **84.71%** | Harmonic mean of class 1 precision and recall |
| **Healthy (Class 0)** | Specificity (TNR) | **100.00%** | 147 correct out of 147 healthy records ($147 / 147$) |
| **Macro Average** | Macro Precision | **89.52%** | Average across Class 0 (79.03%) and Class 1 (100.00%) |
| **Macro Average** | Macro Recall | **86.73%** | Average across Class 0 (100.00%) and Class 1 (73.47%) |
| **Macro Average** | Macro F1-Score | **86.50%** | Average across Class 0 (88.29%) and Class 1 (84.71%) |

> **Dataset and Specificity Note:**
> The dataset `original2.csv` is derived from a public Kaggle dataset ($N = 1,958$, 16 raw columns). After selecting 10 clinical features (dropping `Id`, `Family History`, `Race/Ethnicity`, `Calcium Intake`, and `Medical Conditions`), the Gradient Boosting classifier achieves **73.47%** positive-class sensitivity (108/147) alongside an unusually clean **100.00%** specificity (zero false positives out of 147 healthy records).

---

## Table 4: DenseNet-121 Vision Model Technical Specifications & Audit

| Parameter | Confirmed Value in Repository | Source File | Status / Verification Note |
| :--- | :--- | :--- | :--- |
| **Architecture** | DenseNet-121 | `convert_to_onnx.py:L8` | Verified: `models.densenet121()` |
| **Pretrained Weights** | ImageNet Transfer Learning | `model_loader.py:L59-L61` | Verified: Normalization uses ImageNet mean `[0.485, 0.456, 0.406]` & std `[0.229, 0.224, 0.225]` |
| **Input Dimensions** | 224 × 224 × 3 (RGB) | `model_loader.py:L46-L53` | Input resized to 256×256, center-cropped to 224×224 |
| **Classifier Head (Deployed App)** | Dropout(0.5) → Linear(1024, 256) → ReLU → Dropout(0.5) → Linear(256, 3) | `convert_to_onnx.py:L10-L16` | Custom 3-class classification head in deployed app export |
| **Classifier Head (Kaggle Training)** | Dropout(0.3) → Linear(1024, 256) → ReLU → Dropout(0.2) → Linear(256, 3) | Kaggle Notebook specification | Note: Deployed checkpoint configuration differs in dropout settings |
| **Output Classes** | `['Normal', 'Osteopenia', 'Osteoporosis']` | `model_loader.py:L14` | 3 output classes |
| **Model Weights File** | `densenet121_best.pth` (29.5 MB), `densenet121.onnx` (29.2 MB) | `models/` | Checkpoint is an `OrderedDict` of 729 tensor weights |
| **Training Dataset Name** | **NOT FOUND** | Whole repository | No dataset manifest or training script present in repository |
| **Training Image Counts** | **NOT FOUND** | Whole repository | Raw images and per-class splits not preserved in repository |
| **Number of Epochs** | **NOT FOUND** | Checkpoint inspection | Checkpoint contains no `'epoch'` metadata |
| **Batch Size & LR** | **NOT FOUND** | Checkpoint inspection | Optimizer state stripped during export |
| **Test Accuracy & AUC** | **NOT FOUND** | Whole repository | Standalone test logs not preserved in repository; see perceptual hash audit |
| **Target Anatomical Site** | UI says **"Knee X-Ray Scan"** | `PatientDashboard.jsx:L326` | Updated to "Knee X-Ray Scan" matching knee radiograph training domain |

---

## Table 5: Feature Importances — ExtraTrees DEXA Model (Top 10)

Derived from `ExtraTreesClassifier` trained on `original.csv` ($N = 1,229$ training patients, 20 features).

| Rank | Feature Identifier | Clinical Description | Gini Importance Weight | Cumulative Importance |
| :---: | :--- | :--- | :---: | :---: |
| 1 | **FNT** | Femoral Neck T-Score (DEXA) | **0.1878** | 18.78% |
| 2 | **L1.4T** | Lumbar Spine (L1–L4) T-Score (DEXA) | **0.1784** | 36.62% |
| 3 | **Age** | Patient Age (Years) | **0.0815** | 44.77% |
| 4 | **BMI** | Body Mass Index (kg/m²) | **0.0632** | 51.09% |
| 5 | **Weight** | Patient Weight (kg) | **0.0541** | 56.50% |
| 6 | **Height** | Patient Height (cm) | **0.0498** | 61.48% |
| 7 | **Ca** | Serum Calcium (mg/dL) | **0.0465** | 66.13% |
| 8 | **URIC** | Serum Uric Acid (mg/dL) | **0.0442** | 70.55% |
| 9 | **FBG** | Fasting Blood Glucose (mg/dL) | **0.0423** | 74.78% |
| 10 | **LDL-C** | Low-Density Lipoprotein Cholesterol (mg/dL) | **0.0411** | 78.89% |

---

## Table 6: Master Comparative Summary Across All Systems

| Diagnostic System | Modality | Primary Model | Test Accuracy | Precision | Recall (Sens) | Specificity | F1-Score | ROC-AUC | Status / Caveat |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DEXA Leaky Benchmark** | Tabular (20 features) | ExtraTrees | 88.92% | 86.47% | 92.27% | 85.57% | 89.28% | 0.9462 | **Invalidated by Pre-Split Oversampling Leakage** |
| **DEXA Validated (Full)** | Tabular (20 features) | ExtraTrees | **79.22%** | **79.07%** | **59.65%** | **90.72%** | **68.00%** | **0.8748** | **Leakage-Safe Benchmark on Real Cohort ($N=308$)** |
| **DEXA Validated (No T-Scores)** | Tabular (18 features) | ExtraTrees | **70.45%** | **70.91%** | **34.21%** | **91.75%** | **46.15%** | **0.7177** | **Non-Invasive Baseline (Drops Sensitivity)** |
| **Clinical Triage** | Tabular (10 of 16 features) | Gradient Boosting | **86.73%** | **100.00%** | **73.47%** | **100.00%** | **84.71%** | **0.8624** | **Benchmark on Public Kaggle Dataset ($N=294$)** |
| **Radiographic X-Ray** | Medical Imaging | DenseNet-121 | **NOT FOUND** | **NOT FOUND** | **NOT FOUND** | **NOT FOUND** | **NOT FOUND** | **NOT FOUND** | **Weights Present; System Component with No Benchmark Claim** |

---

## Table 7: Clinical Decision Threshold Tuning & Rule Comparison (Untouched Test Set, $N = 308$)

Comparison of clinical diagnostic cutoff rules, the standard 20-feature ExtraTrees classifier ($\tau = 0.5000$), and the screening-calibrated ExtraTrees classifier ($\tau = 0.3795$, chosen on the training fold via 5-fold CV to target sensitivity $\ge 80\%$).

| Strategy | Decision Rule / Threshold | Accuracy | Sensitivity (Recall) | Specificity (TNR) | Precision (PPV) | F1-Score | Confusion Matrix (TN, FP, FN, TP) | Clinical Role & Trade-off |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **WHO Rule (2 T-Scores)** | $\min(\text{FNT}, \text{L1.4T}) \le -2.5$ | 69.81% | 37.72% (43/114) | **88.66%** (172/194) | 66.15% (43/65) | 48.04% | [172, 22, 71, 43] | High specificity; misses 62.3% of cases ($FN=71$) |
| **WHO Rule (3 T-Scores)** | $\min(\text{FNT}, \text{L1.4T}, \text{TLT}) \le -2.5$ | 70.45% | 39.47% (45/114) | **88.66%** (172/194) | 67.16% (45/67) | 49.72% | [172, 22, 69, 45] | Adds total hip; still misses 60.5% of cases ($FN=69$) |
| **ExtraTrees (Default)** | $\tau = 0.5000$ | 79.22% | 59.65% (68/114) | **90.72%** (176/194) | **79.07%** (68/86) | 68.00% | [176, 18, 46, 68] | Balanced classification; conservative recall |
| **ExtraTrees (Screening)** | **$\tau = 0.3795$** | **81.49%** | **78.07%** (89/114) | 83.51% (162/194) | 73.55% (89/121) | **75.74%** | [162, 32, 25, 89] | **Screening priority: +18.42 pp recall at cost of +14 false positives (18 to 32)** |

> **Diagnostic Label Findings & Dataverse Source:**
> - **Source Deposit:** Harvard Dataverse, author Linfeng He, title *"Bone mineral density"*, DOI: [10.7910/DVN/UDZIJS](https://doi.org/10.7910/DVN/UDZIJS) ($N = 1,537$).
> - **Divergence from T-Scores:** The ground-truth $OP$ label **does not correspond to a T-score threshold**:
>   - In the full dataset, only **41.73%** (237/568) of $OP=1$ cases have $\min(\text{FNT}, \text{L1.4T}, \text{TLT}) \le -2.5$ (58.27% of positive cases have all three site T-scores $>-2.5$).
>   - Conversely, **12.28%** (119/969) of $OP=0$ cases have $\min(\text{FNT}, \text{L1.4T}, \text{TLT}) \le -2.5$.
> - **Paired McNemar Test on 308 Test Patients:**
>   - ExtraTrees at the screening threshold ($\tau = 0.3795$) achieves statistically significant **higher sensitivity** than the 3-T-score rule on the 114 positive patients: ExtraTrees caught 48 true positive cases missed by the rule, while the rule caught only 4 true positive cases missed by ExtraTrees (discordant pairs = 52, continuity-corrected $\chi^2 = 35.56, p = 2.48 \times 10^{-9}, p < 0.0001$).
>   - **Trade-off:** This gain in sensitivity is not without cost: false positives increased from 18 to 32, and specificity decreased from 90.72% to 83.51%.

---

## Table 8: 95% Bootstrap Confidence Intervals (1,000 Resamples)

Evaluated on the untouched test cohort ($N = 308$ patients, 114 positive, 194 negative) for the 20-feature ExtraTrees classifier.

| Evaluation Metric | Point Estimate | 95% Bootstrap Confidence Interval | Standard Error | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **79.22%** | **[74.68%, 83.44%]** | 2.22% | Reliable overall predictive capability |
| **Sensitivity (Recall)** | **59.65%** | **[50.46%, 68.30%]** | 4.54% | True detection rate of osteoporotic patients |
| **Specificity (TNR)** | **90.72%** | **[86.43%, 94.77%]** | 2.11% | Strong rule-out ability for healthy individuals |
| **ROC-AUC** | **0.8748** | **[0.8360, 0.9149]** | 2.01% | High diagnostic discrimination across all thresholds |


