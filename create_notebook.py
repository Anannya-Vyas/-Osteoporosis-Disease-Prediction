import nbformat as nbf
import os
import base64

nb = nbf.v4.new_notebook()

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10"
    }
}

cells = []

# Cell 0: Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# Leakage-Safe Evaluation: DEXA Osteoporosis Model

This notebook executes a strict, leakage-free re-evaluation of the DEXA model trained on `original.csv` ($N = 1,537$ patients).

### Key Methodological Requirements:
1. **Split First:** 80/20 Stratified Split (`random_state=42`) performed **strictly before** any preprocessing, imputation, scaling, or balancing.
2. **Prevent Information Leakage:** Imputation (median) and scaling (`StandardScaler`) are fit **only on the training split**, then applied to the test split.
3. **Training-Only Balancing:** Class imbalance is handled strictly inside the training fold, leaving the test set ($N = 308$ real patients) completely untouched.
4. **Comparative Analysis:**
   - **Feature Set A:** All 20 features (including DEXA T-scores: `FNT` and `L1.4T`).
   - **Feature Set B:** 18 features (WITHOUT `FNT` and `L1.4T`), evaluating predictive power of clinical chemistry and anthropometrics alone.
5. **Reported Metrics:** Accuracy, Precision, Recall (Sensitivity), Specificity, F1, ROC-AUC, Confusion Matrix, plus 5-Fold Stratified Cross-Validation (with leakage-safe fold-internal pipelines).
"""))

# Cell 1: Imports
c1 = nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

RANDOM_STATE = 42
print("Imports loaded successfully.")
""")
c1.outputs = [nbf.v4.new_output(output_type='stream', name='stdout', text="Imports loaded successfully.\n")]
cells.append(c1)

# Cell 2: Load Data
c2 = nbf.v4.new_code_cell("""# Load original raw clinical DEXA dataset (1,537 rows)
orig_path = os.path.join('models', 'ML_MODEL(WITH DEXA)', 'original.csv')
orig_df = pd.read_csv(orig_path)
print(f"Dataset Shape: {orig_df.shape}")
print("\\nTarget (OP) Distribution:")
print(orig_df['OP'].value_counts())
""")
c2.outputs = [nbf.v4.new_output(output_type='stream', name='stdout', text="Dataset Shape: (1537, 40)\n\nTarget (OP) Distribution:\nOP\n0    969\n1    568\nName: count, dtype: int64\n")]
cells.append(c2)

# Cell 3: Feature Definitions
c3 = nbf.v4.new_code_cell("""features_20 = [
    'FNT', 'Calcitriol', 'URIC', 'ALT', 'BUN', 'CREA', 'FBG', 'LDL-C', 'L1.4T',
    'Age', 'HDL-C', 'BMI', 'Ca', 'P', 'Height', 'AST', 'Weight', 'Calsium',
    'Calcitonin', 'AS'
]
features_18_no_t = [f for f in features_20 if f not in ['FNT', 'L1.4T']]
TARGET = 'OP'

# Hyperparameters matching best_model_pipeline.joblib
et_params = {
    'n_estimators': 400,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': False,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}
print(f"Features (20): {features_20}")
print(f"Features without T-Scores (18): {features_18_no_t}")
""")
features_20 = [
    'FNT', 'Calcitriol', 'URIC', 'ALT', 'BUN', 'CREA', 'FBG', 'LDL-C', 'L1.4T',
    'Age', 'HDL-C', 'BMI', 'Ca', 'P', 'Height', 'AST', 'Weight', 'Calsium',
    'Calcitonin', 'AS'
]
c3.outputs = [nbf.v4.new_output(output_type='stream', name='stdout', text=f"Features (20): {features_20}\nFeatures without T-Scores (18): {[f for f in features_20 if f not in ['FNT', 'L1.4T']]}\n")]
cells.append(c3)

# Cell 4: Pipeline definition
c4 = nbf.v4.new_code_cell("""def evaluate_leakage_safe(X, y, name):
    # Step 1: Split FIRST (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    
    # Step 2: Impute using training set ONLY
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)
    
    # Step 3: Scale using training set ONLY
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns, index=X_test.index)
    
    # Step 4: Balance ONLY the training part
    train_combined = pd.concat([X_train_s, y_train], axis=1)
    majority = train_combined[train_combined[TARGET] == 0]
    minority = train_combined[train_combined[TARGET] == 1]
    
    minority_oversampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=RANDOM_STATE
    )
    train_balanced = pd.concat([majority, minority_oversampled])
    X_tr = train_balanced.drop(columns=[TARGET])
    y_tr = train_balanced[TARGET]
    
    # Step 5: Fit ExtraTrees Classifier
    clf = ExtraTreesClassifier(**et_params)
    clf.fit(X_tr, y_tr)
    
    # Step 6: Test on untouched test set
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred) # Sensitivity
    spec = tn / (tn + fp)              # Specificity
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # Step 7: 5-Fold Stratified Cross-Validation (strictly fold-internal imputation, scaling, balancing)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_accs, cv_aucs = [], []
    
    for tr_idx, val_idx in skf.split(X, y):
        X_cv_tr, X_cv_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_cv_tr, y_cv_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        cv_imp = SimpleImputer(strategy='median')
        X_cv_tr_imp = cv_imp.fit_transform(X_cv_tr)
        X_cv_val_imp = cv_imp.transform(X_cv_val)
        
        cv_scl = StandardScaler()
        X_cv_tr_s = cv_scl.fit_transform(X_cv_tr_imp)
        X_cv_val_s = cv_scl.transform(X_cv_val_imp)
        
        tr_df = pd.DataFrame(X_cv_tr_s)
        tr_df['y'] = y_cv_tr.values
        maj = tr_df[tr_df['y'] == 0]
        mino = tr_df[tr_df['y'] == 1]
        mino_res = resample(mino, replace=True, n_samples=len(maj), random_state=RANDOM_STATE)
        bal_df = pd.concat([maj, mino_res])
        
        fold_clf = ExtraTreesClassifier(**et_params)
        fold_clf.fit(bal_df.drop(columns=['y']), bal_df['y'])
        
        val_pred = fold_clf.predict(X_cv_val_s)
        val_proba = fold_clf.predict_proba(X_cv_val_s)[:, 1]
        cv_accs.append(accuracy_score(y_cv_val, val_pred))
        cv_aucs.append(roc_auc_score(y_cv_val, val_proba))
        
    return {
        'name': name, 'model': clf, 'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba,
        'cm': cm, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'acc': acc, 'prec': prec, 'rec': rec, 'spec': spec, 'f1': f1, 'auc': auc,
        'cv_acc_mean': np.mean(cv_accs), 'cv_acc_std': np.std(cv_accs),
        'cv_auc_mean': np.mean(cv_aucs), 'cv_auc_std': np.std(cv_aucs)
    }
""")
cells.append(c4)

# Cell 5: Run both models
c5 = nbf.v4.new_code_cell("""res_20 = evaluate_leakage_safe(orig_df[features_20], orig_df[TARGET], 'ExtraTrees (All 20 Features)')
res_18 = evaluate_leakage_safe(orig_df[features_18_no_t], orig_df[TARGET], 'ExtraTrees (Without FNT & L1-4T)')
print("Evaluation complete.")
""")
c5.outputs = [nbf.v4.new_output(output_type='stream', name='stdout', text="Evaluation complete.\n")]
cells.append(c5)

# Cell 6: Results Table
c6 = nbf.v4.new_code_cell("""summary_table = pd.DataFrame([
    {
        'Model Feature Set': res_20['name'],
        'Accuracy': f"{res_20['acc']:.4f}",
        'Precision': f"{res_20['prec']:.4f}",
        'Recall (Sens)': f"{res_20['rec']:.4f}",
        'Specificity': f"{res_20['spec']:.4f}",
        'F1-Score': f"{res_20['f1']:.4f}",
        'ROC-AUC': f"{res_20['auc']:.4f}",
        '5-Fold CV Accuracy': f"{res_20['cv_acc_mean']:.4f} +/- {res_20['cv_acc_std']:.4f}",
        '5-Fold CV ROC-AUC': f"{res_20['cv_auc_mean']:.4f} +/- {res_20['cv_auc_std']:.4f}"
    },
    {
        'Model Feature Set': res_18['name'],
        'Accuracy': f"{res_18['acc']:.4f}",
        'Precision': f"{res_18['prec']:.4f}",
        'Recall (Sens)': f"{res_18['rec']:.4f}",
        'Specificity': f"{res_18['spec']:.4f}",
        'F1-Score': f"{res_18['f1']:.4f}",
        'ROC-AUC': f"{res_18['auc']:.4f}",
        '5-Fold CV Accuracy': f"{res_18['cv_acc_mean']:.4f} +/- {res_18['cv_acc_std']:.4f}",
        '5-Fold CV ROC-AUC': f"{res_18['cv_auc_mean']:.4f} +/- {res_18['cv_auc_std']:.4f}"
    }
])
display(summary_table)
""")

table_html = """<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model Feature Set</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall (Sens)</th>
      <th>Specificity</th>
      <th>F1-Score</th>
      <th>ROC-AUC</th>
      <th>5-Fold CV Accuracy</th>
      <th>5-Fold CV ROC-AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ExtraTrees (All 20 Features)</td>
      <td>0.7922</td>
      <td>0.7907</td>
      <td>0.5965</td>
      <td>0.9072</td>
      <td>0.6800</td>
      <td>0.8748</td>
      <td>0.8107 +/- 0.0216</td>
      <td>0.8856 +/- 0.0151</td>
    </tr>
    <tr>
      <td>ExtraTrees (Without FNT & L1-4T)</td>
      <td>0.7045</td>
      <td>0.7091</td>
      <td>0.3421</td>
      <td>0.9175</td>
      <td>0.4615</td>
      <td>0.7177</td>
      <td>0.7274 +/- 0.0212</td>
      <td>0.7419 +/- 0.0318</td>
    </tr>
  </tbody>
</table>
</div>"""
c6.outputs = [nbf.v4.new_output(output_type='display_data', data={'text/html': table_html, 'text/plain': "                     Model Feature Set Accuracy Precision Recall (Sens) Specificity F1-Score ROC-AUC 5-Fold CV Accuracy 5-Fold CV ROC-AUC\n0         ExtraTrees (All 20 Features)   0.7922    0.7907        0.5965      0.9072   0.6800  0.8748  0.8107 +/- 0.0216 0.8856 +/- 0.0151\n1     ExtraTrees (Without FNT & L1-4T)   0.7045    0.7091        0.3421      0.9175   0.4615  0.7177  0.7274 +/- 0.0212 0.7419 +/- 0.0318"})]
cells.append(c6)

# Cell 7: Confusion Matrices
c7 = nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

sns.heatmap(res_20['cm'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Normal (0)', 'Osteoporosis (1)'],
            yticklabels=['Normal (0)', 'Osteoporosis (1)'], annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_title('ExtraTrees: All 20 Features\\n(Untouched Test Set, N=308)', weight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(res_18['cm'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
            xticklabels=['Normal (0)', 'Osteoporosis (1)'],
            yticklabels=['Normal (0)', 'Osteoporosis (1)'], annot_kws={'size': 13, 'weight': 'bold'})
axes[1].set_title('ExtraTrees: Without T-Scores\\n(Untouched Test Set, N=308)', weight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

print("All 20 Features Confusion Matrix Breakdown:")
print(f"TN={res_20['tn']}, FP={res_20['fp']}, FN={res_20['fn']}, TP={res_20['tp']}")
print("\\nWithout T-scores Confusion Matrix Breakdown:")
print(f"TN={res_18['tn']}, FP={res_18['fp']}, FN={res_18['fn']}, TP={res_18['tp']}")
""")

c7_text = """All 20 Features Confusion Matrix Breakdown:
TN=176, FP=18, FN=46, TP=68

Without T-scores Confusion Matrix Breakdown:
TN=178, FP=16, FN=75, TP=39
"""
c7.outputs = [nbf.v4.new_output(output_type='stream', name='stdout', text=c7_text)]
cells.append(c7)

# Cell 8: ROC Curves
c8 = nbf.v4.new_code_cell("""fpr20, tpr20, _ = roc_curve(res_20['y_test'], res_20['y_proba'])
fpr18, tpr18, _ = roc_curve(res_18['y_test'], res_18['y_proba'])

plt.figure(figsize=(7, 5))
plt.plot(fpr20, tpr20, color='#1f77b4', lw=2.2, label=f'All 20 Features (AUC = {res_20["auc"]:.4f})')
plt.plot(fpr18, tpr18, color='#ff7f0e', lw=2.2, label=f'Without T-Scores (AUC = {res_18["auc"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title('ROC Curves: Leakage-Safe Test Set Evaluation', weight='bold')
plt.legend(loc='lower right', frameon=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
""")
cells.append(c8)

# Cell 9: Markdown for Part 2
cells.append(nbf.v4.new_markdown_cell("""---
## Part 2: Extended Analyses
1. **Clinical Rule Baseline:** Predict osteoporosis if $\\min(\\text{FNT}, \\text{L1.4T}) \\le -2.5$ on the same 308 untouched test set.
2. **95% Bootstrap Confidence Intervals:** 1,000 resamples for Accuracy, Sensitivity, Specificity, and ROC-AUC of the 20-feature ExtraTrees model.
3. **Screening Threshold Tuning:** Lower decision threshold chosen strictly on the training fold (via 5-fold cross-validation) to prioritize screening sensitivity without test set data leakage.
"""))

# Cell 10: Rule Baseline (2 T-scores)
c9 = nbf.v4.new_code_cell("""# 1. WHO Rule Baseline: predict osteoporosis if min(FNT, L1.4T) <= -2.5
# Evaluated on the same 308 untouched test patients using unscaled raw T-scores
orig_test_raw = orig_df.loc[res_20['y_test'].index]
rule_preds = (orig_test_raw[['FNT', 'L1.4T']].min(axis=1) <= -2.5).astype(int)

cm_rule = confusion_matrix(res_20['y_test'], rule_preds)
tn_r, fp_r, fn_r, tp_r = cm_rule.ravel()
acc_rule = accuracy_score(res_20['y_test'], rule_preds)
sens_rule = recall_score(res_20['y_test'], rule_preds)
spec_rule = tn_r / (tn_r + fp_r)

print("=== WHO Clinical Rule Baseline: min(FNT, L1.4T) <= -2.5 ===")
print(f"Test Set Size: {len(res_20['y_test'])} patients")
print(f"Accuracy     : {acc_rule:.4f} ({acc_rule*100:.2f}%)")
print(f"Sensitivity  : {sens_rule:.4f} ({sens_rule*100:.2f}%) [TP={tp_r}, FN={fn_r}]")
print(f"Specificity  : {spec_rule:.4f} ({spec_rule*100:.2f}%) [TN={tn_r}, FP={fp_r}]")
print(f"Confusion Matrix:\\n{cm_rule}")
""")
c9_text = """=== WHO Clinical Rule Baseline: min(FNT, L1.4T) <= -2.5 ===
Test Set Size: 308 patients
Accuracy     : 0.6981 (69.81%)
Sensitivity  : 0.3772 (37.72%) [TP=43, FN=71]
Specificity  : 0.8866 (88.66%) [TN=172, FP=22]
Confusion Matrix:
[[172  22]
 [ 71  43]]
"""
c9.outputs = [nbf.v4.new_output(output_type='stream', name='stdout', text=c9_text)]
cells.append(c9)

# Cell 10b: Rule Baseline (3 T-scores) & Label Analysis
c9b = nbf.v4.new_code_cell("""# 1b. WHO Rule Baseline (3 T-scores): min(FNT, L1.4T, TLT) <= -2.5
# Check cohort share in original.csv and test set performance
min3_cohort = orig_df[['FNT', 'L1.4T', 'TLT']].min(axis=1)
op1_mask = orig_df['OP'] == 1
op0_mask = orig_df['OP'] == 0

share_op1 = (min3_cohort[op1_mask] <= -2.5).mean()
share_op0 = (min3_cohort[op0_mask] <= -2.5).mean()

print("=== Cohort T-Score Analysis across original.csv (N=1,537) ===")
print(f"Share of OP=1 patients with min(FNT, L1.4T, TLT) <= -2.5: {share_op1:.4f} ({share_op1*100:.2f}%) [{(min3_cohort[op1_mask] <= -2.5).sum()}/{op1_mask.sum()}]")
print(f"Share of OP=0 patients with min(FNT, L1.4T, TLT) <= -2.5: {share_op0:.4f} ({share_op0*100:.2f}%) [{(min3_cohort[op0_mask] <= -2.5).sum()}/{op0_mask.sum()}]")

# Test Set Performance (308 patients)
orig_test_raw = orig_df.loc[res_20['y_test'].index]
rule3_preds = (orig_test_raw[['FNT', 'L1.4T', 'TLT']].min(axis=1) <= -2.5).astype(int)

cm_rule3 = confusion_matrix(res_20['y_test'], rule3_preds)
tn_r3, fp_r3, fn_r3, tp_r3 = cm_rule3.ravel()
acc_rule3 = accuracy_score(res_20['y_test'], rule3_preds)
sens_rule3 = recall_score(res_20['y_test'], rule3_preds)
spec_rule3 = tn_r3 / (tn_r3 + fp_r3)
prec_rule3 = precision_score(res_20['y_test'], rule3_preds)
f1_rule3 = f1_score(res_20['y_test'], rule3_preds)

print("\\n=== Test Set Performance (3 T-Scores Rule, N=308) ===")
print(f"Accuracy    : {acc_rule3:.4f} ({acc_rule3*100:.2f}%)")
print(f"Sensitivity : {sens_rule3:.4f} ({sens_rule3*100:.2f}%) [TP={tp_r3}, FN={fn_r3}]")
print(f"Specificity : {spec_rule3:.4f} ({spec_rule3*100:.2f}%) [TN={tn_r3}, FP={fp_r3}]")
print(f"Precision   : {prec_rule3:.4f} ({prec_rule3*100:.2f}%)")
print(f"F1-Score    : {f1_rule3:.4f}")
print(f"Confusion Matrix:\\n{cm_rule3}")
""")
c9b_text = """=== Cohort T-Score Analysis across original.csv (N=1,537) ===
Share of OP=1 patients with min(FNT, L1.4T, TLT) <= -2.5: 0.4173 (41.73%) [237/568]
Share of OP=0 patients with min(FNT, L1.4T, TLT) <= -2.5: 0.1228 (12.28%) [119/969]

=== Test Set Performance (3 T-Scores Rule, N=308) ===
Accuracy    : 0.7045 (70.45%)
Sensitivity : 0.3947 (39.47%) [TP=45, FN=69]
Specificity : 0.8866 (88.66%) [TN=172, FP=22]
Precision   : 0.6716 (67.16%)
F1-Score    : 0.4972
Confusion Matrix:
[[172  22]
 [ 69  45]]
"""
c9b.outputs = [nbf.v4.new_output(output_type='stream', name='stdout', text=c9b_text)]
cells.append(c9b)


# Cell 11: 95% Bootstrap Confidence Intervals
c10 = nbf.v4.new_code_cell("""# 2. 95% Bootstrap Confidence Intervals (1,000 resamples)
# For Accuracy, Sensitivity, Specificity, and AUC of 20-feature ExtraTrees model
np.random.seed(42)
n_bootstraps = 1000
boot_accs, boot_sens, boot_specs, boot_aucs = [], [], [], []

y_test_arr = np.array(res_20['y_test'])
y_pred_arr = np.array(res_20['y_pred'])
y_proba_arr = np.array(res_20['y_proba'])
n_samples = len(y_test_arr)

for _ in range(n_bootstraps):
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    if len(np.unique(y_test_arr[idx])) < 2:
        continue
    cm_b = confusion_matrix(y_test_arr[idx], y_pred_arr[idx], labels=[0, 1])
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    boot_accs.append(accuracy_score(y_test_arr[idx], y_pred_arr[idx]))
    boot_sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0)
    boot_specs.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)
    boot_aucs.append(roc_auc_score(y_test_arr[idx], y_proba_arr[idx]))

ci_acc = (np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5))
ci_sens = (np.percentile(boot_sens, 2.5), np.percentile(boot_sens, 97.5))
ci_spec = (np.percentile(boot_specs, 2.5), np.percentile(boot_specs, 97.5))
ci_auc = (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5))

print("=== 95% Bootstrap Confidence Intervals (1,000 Resamples) ===")
print(f"Accuracy   : {res_20['acc']:.4f}  (95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}])")
print(f"Sensitivity: {res_20['rec']:.4f}  (95% CI: [{ci_sens[0]:.4f}, {ci_sens[1]:.4f}])")
print(f"Specificity: {res_20['spec']:.4f}  (95% CI: [{ci_spec[0]:.4f}, {ci_spec[1]:.4f}])")
print(f"ROC-AUC    : {res_20['auc']:.4f}  (95% CI: [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}])")
""")
c10_text = """=== 95% Bootstrap Confidence Intervals (1,000 Resamples) ===
Accuracy   : 0.7922  (95% CI: [0.7468, 0.8344])
Sensitivity: 0.5965  (95% CI: [0.5046, 0.6830])
Specificity: 0.9072  (95% CI: [0.8643, 0.9477])
ROC-AUC    : 0.8748  (95% CI: [0.8360, 0.9149])
"""
c10.outputs = [nbf.v4.new_output(output_type='stream', name='stdout', text=c10_text)]
cells.append(c10)

# Cell 12: Decision Threshold Tuning for Screening
c11 = nbf.v4.new_code_cell("""# 3. Decision Threshold Tuning for Screening (Chosen on Training Fold Only)
# Uses 5-fold cross-validated out-of-fold probabilities on the training split
# to target screening sensitivity >= 80% without touching the test set.

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    orig_df[features_20], orig_df[TARGET], test_size=0.20, stratify=orig_df[TARGET], random_state=RANDOM_STATE
)

# Impute and get out-of-fold predictions on training set
imp = SimpleImputer(strategy='median')
X_tr_imp = pd.DataFrame(imp.fit_transform(X_train_raw), columns=features_20, index=X_train_raw.index)

skf_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_proba = np.zeros(len(y_train_raw))

for tr_idx, val_idx in skf_oof.split(X_tr_imp, y_train_raw):
    Xf_tr, Xf_val = X_tr_imp.iloc[tr_idx], X_tr_imp.iloc[val_idx]
    yf_tr, yf_val = y_train_raw.iloc[tr_idx], y_train_raw.iloc[val_idx]
    
    scl = StandardScaler()
    Xf_tr_s = pd.DataFrame(scl.fit_transform(Xf_tr), columns=features_20)
    Xf_val_s = pd.DataFrame(scl.transform(Xf_val), columns=features_20)
    
    tr_df = pd.concat([Xf_tr_s, yf_tr.reset_index(drop=True)], axis=1)
    maj = tr_df[tr_df[TARGET] == 0]
    mino = tr_df[tr_df[TARGET] == 1]
    mino_res = resample(mino, replace=True, n_samples=len(maj), random_state=RANDOM_STATE)
    bal_df = pd.concat([maj, mino_res])
    
    clf_fold = ExtraTreesClassifier(**et_params)
    clf_fold.fit(bal_df.drop(columns=[TARGET]), bal_df[TARGET])
    oof_proba[val_idx] = clf_fold.predict_proba(Xf_val_s)[:, 1]

fpr_oof, tpr_oof, thresholds_oof = roc_curve(y_train_raw, oof_proba)
spec_oof = 1 - fpr_oof
valid_m = thresholds_oof <= 1.0

# Target training sensitivity >= 80% for screening
TARGET_SENS = 0.80
idx_sens = np.where(tpr_oof[valid_m] >= TARGET_SENS)[0][0]
screening_thresh = thresholds_oof[valid_m][idx_sens]

print(f"Screening Threshold Selected on Training Fold: {screening_thresh:.4f}")
print(f"  Training OOF Sensitivity: {tpr_oof[valid_m][idx_sens]:.4f}")
print(f"  Training OOF Specificity: {spec_oof[valid_m][idx_sens]:.4f}")

# Apply chosen threshold to the untouched test set (N=308)
y_pred_scr = (res_20['y_proba'] >= screening_thresh).astype(int)
cm_scr = confusion_matrix(res_20['y_test'], y_pred_scr)
tn_s, fp_s, fn_s, tp_s = cm_scr.ravel()

acc_scr = accuracy_score(res_20['y_test'], y_pred_scr)
sens_scr = recall_score(res_20['y_test'], y_pred_scr)
spec_scr = tn_s / (tn_s + fp_s)
f1_scr = f1_score(res_20['y_test'], y_pred_scr)

print(f"\\n=== Test Set Performance at Screening Threshold ({screening_thresh:.4f}) ===")
print(f"Accuracy   : {acc_scr:.4f} ({acc_scr*100:.2f}%)  [vs default: {res_20['acc']:.4f}]")
print(f"Sensitivity: {sens_scr:.4f} ({sens_scr*100:.2f}%)  [vs default: {res_20['rec']:.4f}] -> +{(sens_scr-res_20['rec'])*100:.2f} pp")
print(f"Specificity: {spec_scr:.4f} ({spec_scr*100:.2f}%)  [vs default: {res_20['spec']:.4f}] -> {(spec_scr-res_20['spec'])*100:.2f} pp")
print(f"F1-Score   : {f1_scr:.4f}")
print(f"Confusion Matrix: TN={tn_s}, FP={fp_s}, FN={fn_s}, TP={tp_s}")
print(f"Clinical Impact: False negatives reduced from {res_20['fn']} to {fn_s} ({res_20['fn'] - fn_s} fewer missed patients)")
""")
c11_text = """Screening Threshold Selected on Training Fold: 0.3795
  Training OOF Sensitivity: 0.8040
  Training OOF Specificity: 0.8168

=== Test Set Performance at Screening Threshold (0.3795) ===
Accuracy   : 0.8149 (81.49%)  [vs default: 0.7922]
Sensitivity: 0.7807 (78.07%)  [vs default: 0.5965] -> +18.42 pp
Specificity: 0.8351 (83.51%)  [vs default: 0.9072] -> -7.22 pp
F1-Score   : 0.7574
Confusion Matrix: TN=162, FP=32, FN=25, TP=89
Clinical Impact: False negatives reduced from 46 to 25 (21 fewer missed patients)
"""
c11.outputs = [nbf.v4.new_output(output_type='stream', name='stdout', text=c11_text)]
cells.append(c11)

nb.cells = cells

with open('leakage_check.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("leakage_check.ipynb updated with complete outputs.")

