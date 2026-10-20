"""
Extended Analysis for leakage_check.ipynb
==========================================
1. WHO Rule Baseline: predict osteoporosis if min(FNT, L1.4T) <= -2.5
2. 95% Bootstrap CIs (1000 resamples) for accuracy, sensitivity, specificity, AUC
3. Threshold tuning on TRAINING fold only, favouring screening sensitivity

All figures saved at 300 dpi in paper_figures/.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample as sk_resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

RANDOM_STATE = 42
FIGURES_DIR = 'paper_figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)

# -------------------------------------------------------------------
# Load data & reproduce the exact same 80/20 split as the main notebook
# -------------------------------------------------------------------
orig_path = os.path.join('models', 'ML_MODEL(WITH DEXA)', 'original.csv')
orig_df = pd.read_csv(orig_path)
print(f"Dataset loaded: {orig_df.shape[0]} rows, {orig_df.shape[1]} columns")

features_20 = [
    'FNT', 'Calcitriol', 'URIC', 'ALT', 'BUN', 'CREA', 'FBG', 'LDL-C', 'L1.4T',
    'Age', 'HDL-C', 'BMI', 'Ca', 'P', 'Height', 'AST', 'Weight', 'Calsium',
    'Calcitonin', 'AS'
]
TARGET = 'OP'

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

X = orig_df[features_20]
y = orig_df[TARGET]

# Step 1: Split FIRST (identical to leakage_check.ipynb)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Test class distribution: {y_test.value_counts().to_dict()}")

# Step 2: Impute using training set ONLY
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

# Step 3: Scale using training set ONLY
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X.columns, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns, index=X_test.index)

# Step 4: Balance ONLY the training part (oversample minority)
train_combined = pd.concat([X_train_s, y_train], axis=1)
majority = train_combined[train_combined[TARGET] == 0]
minority = train_combined[train_combined[TARGET] == 1]
minority_oversampled = sk_resample(
    minority, replace=True, n_samples=len(majority), random_state=RANDOM_STATE
)
train_balanced = pd.concat([majority, minority_oversampled])
X_tr = train_balanced.drop(columns=[TARGET])
y_tr = train_balanced[TARGET]

# Step 5: Train ExtraTrees
clf = ExtraTreesClassifier(**et_params)
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_test_s)
y_proba = clf.predict_proba(X_test_s)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
sens = recall_score(y_test, y_pred)
spec = tn / (tn + fp)
auc_val = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("BASELINE: 20-feature ExtraTrees on untouched test set")
print("=" * 60)
print(f"Accuracy    : {acc:.4f}")
print(f"Sensitivity : {sens:.4f}")
print(f"Specificity : {spec:.4f}")
print(f"AUC         : {auc_val:.4f}")
print(f"Precision   : {prec:.4f}")
print(f"F1          : {f1:.4f}")
print(f"Confusion   : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ===================================================================
# ANALYSIS 1: WHO Rule Baseline
# Predict osteoporosis if min(FNT, L1.4T) <= -2.5
# Applied on the SAME untouched test set (using raw, un-scaled values)
# ===================================================================
print("\n" + "=" * 60)
print("ANALYSIS 1: WHO Rule Baseline (min(FNT, L1.4T) <= -2.5)")
print("=" * 60)

# Use the raw (unscaled, imputed) test values for the clinical rule
X_test_raw = X_test_imp.copy()  # imputed but NOT scaled - real T-score values
rule_pred = (X_test_raw[['FNT', 'L1.4T']].min(axis=1) <= -2.5).astype(int)

cm_rule = confusion_matrix(y_test, rule_pred)
tn_r, fp_r, fn_r, tp_r = cm_rule.ravel()
acc_rule = accuracy_score(y_test, rule_pred)
sens_rule = recall_score(y_test, rule_pred)
spec_rule = tn_r / (tn_r + fp_r)

print(f"Accuracy    : {acc_rule:.4f} ({acc_rule*100:.2f}%)")
print(f"Sensitivity : {sens_rule:.4f} ({sens_rule*100:.2f}%)")
print(f"Specificity : {spec_rule:.4f} ({spec_rule*100:.2f}%)")
print(f"Confusion   : TN={tn_r}, FP={fp_r}, FN={fn_r}, TP={tp_r}")
print(f"Test N      : {len(y_test)}")

# Also compute PPV and NPV
ppv_rule = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0.0
npv_rule = tn_r / (tn_r + fn_r) if (tn_r + fn_r) > 0 else 0.0
print(f"PPV         : {ppv_rule:.4f}")
print(f"NPV         : {npv_rule:.4f}")

# Figure: Confusion matrix for rule baseline
fig_rule, ax_rule = plt.subplots(figsize=(5, 4.2), dpi=300)
sns.heatmap(cm_rule, annot=True, fmt='d', cmap='Oranges', cbar=False,
            xticklabels=['Normal (0)', 'Osteoporosis (1)'],
            yticklabels=['Normal (0)', 'Osteoporosis (1)'], ax=ax_rule,
            annot_kws={'size': 14, 'weight': 'bold'})
ax_rule.set_title('WHO Rule Baseline: min(FNT, L1.4T) <= -2.5\n(Untouched Test Set, N=308)',
                  fontsize=11, weight='bold', pad=12)
ax_rule.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
ax_rule.set_ylabel('True Label', fontsize=11, labelpad=8)
plt.tight_layout()
fig_rule.savefig(os.path.join(FIGURES_DIR, 'cm_rule_baseline.png'), dpi=300)
plt.close(fig_rule)
print(f"Saved: {os.path.join(FIGURES_DIR, 'cm_rule_baseline.png')}")


# ===================================================================
# ANALYSIS 1B: WHO Rule Baseline (3 T-Scores: FNT, L1.4T, TLT <= -2.5)
# Using raw unscaled values from original.csv
# ===================================================================
print("\n" + "=" * 60)
print("ANALYSIS 1B: WHO Rule Baseline (min(FNT, L1.4T, TLT) <= -2.5)")
print("=" * 60)

# Check across ALL 1,537 rows of original.csv
min3_all = orig_df[['FNT', 'L1.4T', 'TLT']].min(axis=1)
op1_mask = orig_df['OP'] == 1
op0_mask = orig_df['OP'] == 0
count_op1_rule = (min3_all[op1_mask] <= -2.5).sum()
total_op1 = op1_mask.sum()
share_op1 = count_op1_rule / total_op1

count_op0_rule = (min3_all[op0_mask] <= -2.5).sum()
total_op0 = op0_mask.sum()
share_op0 = count_op0_rule / total_op0

print(f"Cohort Analysis across ALL {len(orig_df)} rows in original.csv:")
print(f"  OP=1 patients with min(FNT, L1.4T, TLT) <= -2.5: {count_op1_rule} / {total_op1} ({share_op1*100:.2f}%)")
print(f"  OP=0 patients with min(FNT, L1.4T, TLT) <= -2.5: {count_op0_rule} / {total_op0} ({share_op0*100:.2f}%)")

# Test Set Evaluation (exact same 308 untouched patients)
orig_test_raw = orig_df.loc[y_test.index]
rule3_pred = (orig_test_raw[['FNT', 'L1.4T', 'TLT']].min(axis=1) <= -2.5).astype(int)

cm_rule3 = confusion_matrix(y_test, rule3_pred)
tn_r3, fp_r3, fn_r3, tp_r3 = cm_rule3.ravel()
acc_rule3 = accuracy_score(y_test, rule3_pred)
sens_rule3 = recall_score(y_test, rule3_pred)
spec_rule3 = tn_r3 / (tn_r3 + fp_r3)
prec_rule3 = precision_score(y_test, rule3_pred)
f1_rule3 = f1_score(y_test, rule3_pred)

print(f"\nTest Set Performance (3-T-Score Rule, N={len(y_test)}):")
print(f"  Accuracy    : {acc_rule3:.4f} ({acc_rule3*100:.2f}%)")
print(f"  Sensitivity : {sens_rule3:.4f} ({sens_rule3*100:.2f}%) [TP={tp_r3}, FN={fn_r3}]")
print(f"  Specificity : {spec_rule3:.4f} ({spec_rule3*100:.2f}%) [TN={tn_r3}, FP={fp_r3}]")
print(f"  Precision   : {prec_rule3:.4f} ({prec_rule3*100:.2f}%)")
print(f"  F1-Score    : {f1_rule3:.4f}")
print(f"  Confusion Matrix: TN={tn_r3}, FP={fp_r3}, FN={fn_r3}, TP={tp_r3}")

# Save 300 dpi confusion matrix figure
fig_r3, ax_r3 = plt.subplots(figsize=(5, 4.2), dpi=300)
sns.heatmap(cm_rule3, annot=True, fmt='d', cmap='Oranges', cbar=False,
            xticklabels=['Normal (0)', 'Osteoporosis (1)'],
            yticklabels=['Normal (0)', 'Osteoporosis (1)'], ax=ax_r3,
            annot_kws={'size': 14, 'weight': 'bold'})
ax_r3.set_title('WHO Rule Baseline: min(FNT, L1.4T, TLT) <= -2.5\n(Untouched Test Set, N=308)',
                fontsize=11, weight='bold', pad=12)
ax_r3.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
ax_r3.set_ylabel('True Label', fontsize=11, labelpad=8)
plt.tight_layout()
fig_r3.savefig(os.path.join(FIGURES_DIR, 'cm_rule_3tscores.png'), dpi=300)
plt.close(fig_r3)
print(f"Saved: {os.path.join(FIGURES_DIR, 'cm_rule_3tscores.png')}")



# ===================================================================
# ANALYSIS 2: 95% Bootstrap Confidence Intervals (1000 resamples)
# For accuracy, sensitivity, specificity, and AUC of 20-feature model
# ===================================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: 95% Bootstrap Confidence Intervals (1000 resamples)")
print("=" * 60)

n_boot = 1000
boot_accs = []
boot_sens = []
boot_spec = []
boot_aucs = []

rng = np.random.RandomState(RANDOM_STATE)

y_test_arr = np.array(y_test)
y_pred_arr = np.array(y_pred)
y_proba_arr = np.array(y_proba)

for i in range(n_boot):
    idx = rng.choice(len(y_test_arr), size=len(y_test_arr), replace=True)
    y_t = y_test_arr[idx]
    y_p = y_pred_arr[idx]
    y_pr = y_proba_arr[idx]
    
    # Skip degenerate samples (single class only)
    if len(np.unique(y_t)) < 2:
        continue
    
    cm_b = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    
    boot_accs.append(accuracy_score(y_t, y_p))
    boot_sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0)
    boot_spec.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0)
    boot_aucs.append(roc_auc_score(y_t, y_pr))

boot_accs = np.array(boot_accs)
boot_sens = np.array(boot_sens)
boot_spec = np.array(boot_spec)
boot_aucs = np.array(boot_aucs)

def ci95(arr):
    lo = np.percentile(arr, 2.5)
    hi = np.percentile(arr, 97.5)
    return lo, hi

ci_acc = ci95(boot_accs)
ci_sens = ci95(boot_sens)
ci_spec = ci95(boot_spec)
ci_auc = ci95(boot_aucs)

print(f"Valid bootstrap samples: {len(boot_accs)} / {n_boot}")
print(f"Accuracy    : {acc:.4f}  95% CI [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
print(f"Sensitivity : {sens:.4f}  95% CI [{ci_sens[0]:.4f}, {ci_sens[1]:.4f}]")
print(f"Specificity : {spec:.4f}  95% CI [{ci_spec[0]:.4f}, {ci_spec[1]:.4f}]")
print(f"AUC         : {auc_val:.4f}  95% CI [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")

# Figure: Bootstrap distribution with CIs
fig_boot, axes_boot = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
metrics_data = [
    ('Accuracy', boot_accs, acc, ci_acc),
    ('Sensitivity', boot_sens, sens, ci_sens),
    ('Specificity', boot_spec, spec, ci_spec),
    ('ROC-AUC', boot_aucs, auc_val, ci_auc)
]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for ax, (name, data, point, ci), color in zip(axes_boot.flat, metrics_data, colors):
    ax.hist(data, bins=40, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axvline(point, color='black', linestyle='-', linewidth=2, label=f'Point Est. = {point:.4f}')
    ax.axvline(ci[0], color='red', linestyle='--', linewidth=1.5, label=f'2.5% = {ci[0]:.4f}')
    ax.axvline(ci[1], color='red', linestyle='--', linewidth=1.5, label=f'97.5% = {ci[1]:.4f}')
    ax.set_title(f'{name} Bootstrap Distribution', fontsize=11, weight='bold')
    ax.set_xlabel(name, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.2)

fig_boot.suptitle('95% Bootstrap Confidence Intervals (1,000 Resamples)\nExtraTrees (20 Features) — Untouched Test Set',
                  fontsize=13, weight='bold', y=1.02)
plt.tight_layout()
fig_boot.savefig(os.path.join(FIGURES_DIR, 'bootstrap_ci_distributions.png'), dpi=300, bbox_inches='tight')
plt.close(fig_boot)
print(f"Saved: {os.path.join(FIGURES_DIR, 'bootstrap_ci_distributions.png')}")


# ===================================================================
# ANALYSIS 3: Threshold Tuning on TRAINING Fold
# Use 5-fold cross-validated out-of-fold probabilities on the ORIGINAL
# (pre-balance) training split to find a screening-friendly threshold.
# Goal: maximise sensitivity subject to specificity >= 0.60
# ===================================================================
print("\n" + "=" * 60)
print("ANALYSIS 3: Threshold Tuning for Screening (Training Fold Only)")
print("=" * 60)

# Collect out-of-fold predicted probabilities via 5-fold CV on the
# ORIGINAL training split (X_train_imp/y_train), with fold-internal
# imputation, scaling, and oversampling — no test data involved.
from sklearn.model_selection import StratifiedKFold as SKF_inner

oof_proba = np.full(len(y_train), np.nan)
skf_inner = SKF_inner(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for tr_idx, val_idx in skf_inner.split(X_train_imp, y_train):
    Xf_tr, Xf_val = X_train_imp.iloc[tr_idx], X_train_imp.iloc[val_idx]
    yf_tr, yf_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    scl_f = StandardScaler()
    Xf_tr_s = pd.DataFrame(scl_f.fit_transform(Xf_tr), columns=X.columns)
    Xf_val_s = pd.DataFrame(scl_f.transform(Xf_val), columns=X.columns)

    # Oversample minority inside fold
    tr_df_f = Xf_tr_s.copy()
    tr_df_f['y'] = yf_tr.values
    maj_f = tr_df_f[tr_df_f['y'] == 0]
    min_f = tr_df_f[tr_df_f['y'] == 1]
    min_f_res = sk_resample(min_f, replace=True, n_samples=len(maj_f), random_state=RANDOM_STATE)
    bal_f = pd.concat([maj_f, min_f_res])

    fold_clf = ExtraTreesClassifier(**et_params)
    fold_clf.fit(bal_f.drop(columns=['y']), bal_f['y'])
    oof_proba[val_idx] = fold_clf.predict_proba(Xf_val_s)[:, 1]

# ROC on the out-of-fold training probabilities
fpr_oof, tpr_oof, thresholds_oof = roc_curve(y_train, oof_proba)
specificity_oof = 1 - fpr_oof

# Retain valid thresholds in [0, 1]
valid_mask = thresholds_oof <= 1.0
th_v = thresholds_oof[valid_mask]
tpr_v = tpr_oof[valid_mask]
spec_v = specificity_oof[valid_mask]

# Screening selection strategy on TRAINING fold:
# Target high sensitivity (>= 80% on training fold) with maximal specificity
TARGET_SENS = 0.80
sens_mask = tpr_v >= TARGET_SENS

if sens_mask.any():
    # Pick the highest threshold satisfying sensitivity >= 80%
    chosen_idx = np.where(sens_mask)[0][0]
    screening_threshold = th_v[chosen_idx]
    train_sens_at_thresh = tpr_v[chosen_idx]
    train_spec_at_thresh = spec_v[chosen_idx]
else:
    # Fallback: Weighted Youden Index prioritizing sensitivity (2*Sens + Spec - 1)
    j_weighted = 2 * tpr_v + spec_v - 1
    chosen_idx = np.argmax(j_weighted)
    screening_threshold = th_v[chosen_idx]
    train_sens_at_thresh = tpr_v[chosen_idx]
    train_spec_at_thresh = spec_v[chosen_idx]

print(f"Screening threshold chosen on OOF TRAINING probabilities: {screening_threshold:.4f}")
print(f"  Criterion: Target training sensitivity >= {TARGET_SENS*100:.0f}%")
print(f"  Training (OOF) Sensitivity at threshold: {train_sens_at_thresh:.4f} ({train_sens_at_thresh*100:.2f}%)")
print(f"  Training (OOF) Specificity at threshold: {train_spec_at_thresh:.4f} ({train_spec_at_thresh*100:.2f}%)")

# Apply chosen threshold to the UNTOUCHED test set (N=308)
y_pred_screen = (y_proba >= screening_threshold).astype(int)
cm_screen = confusion_matrix(y_test, y_pred_screen)
tn_s, fp_s, fn_s, tp_s = cm_screen.ravel()

acc_screen = accuracy_score(y_test, y_pred_screen)
sens_screen = recall_score(y_test, y_pred_screen)
spec_screen = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0
f1_screen = f1_score(y_test, y_pred_screen)
prec_screen = precision_score(y_test, y_pred_screen)

print(f"\n--- TEST SET results at screening threshold = {screening_threshold:.4f} ---")
print(f"Accuracy    : {acc_screen:.4f} ({acc_screen*100:.2f}%)")
print(f"Sensitivity : {sens_screen:.4f} ({sens_screen*100:.2f}%)")
print(f"Specificity : {spec_screen:.4f} ({spec_screen*100:.2f}%)")
print(f"Precision   : {prec_screen:.4f} ({prec_screen*100:.2f}%)")
print(f"F1          : {f1_screen:.4f}")
print(f"Confusion   : TN={tn_s}, FP={fp_s}, FN={fn_s}, TP={tp_s}")
print(f"\nComparison with default threshold (0.5000):")
print(f"  Sensitivity: {sens:.4f} -> {sens_screen:.4f}  (change: {(sens_screen-sens)*100:+.2f} percentage points)")
print(f"  Specificity: {spec:.4f} -> {spec_screen:.4f}  (change: {(spec_screen-spec)*100:+.2f} percentage points)")
print(f"  Accuracy   : {acc:.4f} -> {acc_screen:.4f}  (change: {(acc_screen-acc)*100:+.2f} percentage points)")
print(f"  FN caught  : {fn} -> {fn_s}  ({fn - fn_s} fewer missed osteoporotic patients)")

# Figure: Confusion matrix at screening threshold
fig_scr, ax_scr = plt.subplots(figsize=(5, 4.2), dpi=300)
sns.heatmap(cm_screen, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['Normal (0)', 'Osteoporosis (1)'],
            yticklabels=['Normal (0)', 'Osteoporosis (1)'], ax=ax_scr,
            annot_kws={'size': 14, 'weight': 'bold'})
ax_scr.set_title(f'ExtraTrees @ Screening Threshold = {screening_threshold:.3f}\n(Untouched Test Set, N=308)',
                 fontsize=11, weight='bold', pad=12)
ax_scr.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
ax_scr.set_ylabel('True Label', fontsize=11, labelpad=8)
plt.tight_layout()
fig_scr.savefig(os.path.join(FIGURES_DIR, 'cm_screening_threshold.png'), dpi=300)
plt.close(fig_scr)
print(f"Saved: {os.path.join(FIGURES_DIR, 'cm_screening_threshold.png')}")

# Figure: Threshold vs Sensitivity / Specificity trade-off (OOF training probabilities)
fig_tradeoff, ax_to = plt.subplots(figsize=(7, 5), dpi=300)
ax_to.plot(th_v, tpr_v, color='#1f77b4', lw=2.2, label='Sensitivity (OOF Train)', linestyle='-')
ax_to.plot(th_v, spec_v, color='#ff7f0e', lw=2.2, label='Specificity (OOF Train)', linestyle='-')

# Mark the chosen threshold and default threshold
ax_to.axvline(screening_threshold, color='red', linestyle='--', lw=1.8, alpha=0.85,
              label=f'Screening Threshold = {screening_threshold:.3f}')
ax_to.axvline(0.5, color='gray', linestyle=':', lw=1.5, alpha=0.7,
              label='Default Threshold = 0.500')

# Mark test performance at chosen threshold
ax_to.scatter([screening_threshold], [sens_screen], color='red', s=90, zorder=5,
              marker='*', label=f'Test Sens = {sens_screen:.3f}')
ax_to.scatter([screening_threshold], [spec_screen], color='darkred', s=90, zorder=5,
              marker='D', label=f'Test Spec = {spec_screen:.3f}')

ax_to.set_xlabel('Decision Threshold', fontsize=11)
ax_to.set_ylabel('Metric Value', fontsize=11)
ax_to.set_title('Sensitivity vs Specificity Trade-Off\n(Threshold Chosen on OOF Training Probabilities)', fontsize=12, weight='bold', pad=12)
ax_to.legend(fontsize=8.5, loc='center left')
ax_to.set_xlim([-0.02, 1.02])
ax_to.set_ylim([-0.02, 1.05])
ax_to.grid(alpha=0.3)
plt.tight_layout()
fig_tradeoff.savefig(os.path.join(FIGURES_DIR, 'threshold_tradeoff.png'), dpi=300)
plt.close(fig_tradeoff)
print(f"Saved: {os.path.join(FIGURES_DIR, 'threshold_tradeoff.png')}")


# ===================================================================
# SUMMARY TABLE
# ===================================================================
print("\n" + "=" * 60)
print("COMBINED SUMMARY TABLE")
print("=" * 60)

summary = pd.DataFrame([
    {
        'Method': 'WHO Rule (2 T-scores)',
        'Threshold': 'min(FNT, L1.4T) <= -2.5',
        'Accuracy': f'{acc_rule:.4f}',
        'Sensitivity': f'{sens_rule:.4f}',
        'Specificity': f'{spec_rule:.4f}',
        'AUC': 'N/A',
        '95% CI (Sens)': 'N/A'
    },
    {
        'Method': 'WHO Rule (3 T-scores)',
        'Threshold': 'min(FNT, L1.4T, TLT) <= -2.5',
        'Accuracy': f'{acc_rule3:.4f}',
        'Sensitivity': f'{sens_rule3:.4f}',
        'Specificity': f'{spec_rule3:.4f}',
        'AUC': 'N/A',
        '95% CI (Sens)': 'N/A'
    },
    {
        'Method': 'ExtraTrees (20 feat)',
        'Threshold': '0.5000',
        'Accuracy': f'{acc:.4f}',
        'Sensitivity': f'{sens:.4f}',
        'Specificity': f'{spec:.4f}',
        'AUC': f'{auc_val:.4f}',
        '95% CI (Sens)': f'[{ci_sens[0]:.4f}, {ci_sens[1]:.4f}]'
    },
    {
        'Method': 'ExtraTrees (screening)',
        'Threshold': f'{screening_threshold:.4f}',
        'Accuracy': f'{acc_screen:.4f}',
        'Sensitivity': f'{sens_screen:.4f}',
        'Specificity': f'{spec_screen:.4f}',
        'AUC': f'{auc_val:.4f}',
        '95% CI (Sens)': 'N/A'
    }
])
print(summary.to_string(index=False))

# Save summary as JSON for notebook consumption
results_dict = {
    'rule_baseline_2t': {
        'accuracy': float(acc_rule),
        'sensitivity': float(sens_rule),
        'specificity': float(spec_rule),
        'ppv': float(ppv_rule),
        'npv': float(npv_rule),
        'cm': {'tn': int(tn_r), 'fp': int(fp_r), 'fn': int(fn_r), 'tp': int(tp_r)}
    },
    'rule_baseline_3t': {
        'accuracy': float(acc_rule3),
        'sensitivity': float(sens_rule3),
        'specificity': float(spec_rule3),
        'ppv': float(prec_rule3),
        'f1': float(f1_rule3),
        'share_op1_cohort': float(share_op1),
        'share_op0_cohort': float(share_op0),
        'cm': {'tn': int(tn_r3), 'fp': int(fp_r3), 'fn': int(fn_r3), 'tp': int(tp_r3)}
    },
    'bootstrap_ci': {
        'n_resamples': n_boot,
        'valid_samples': len(boot_accs),
        'accuracy': {'point': float(acc), 'ci_lower': float(ci_acc[0]), 'ci_upper': float(ci_acc[1])},
        'sensitivity': {'point': float(sens), 'ci_lower': float(ci_sens[0]), 'ci_upper': float(ci_sens[1])},
        'specificity': {'point': float(spec), 'ci_lower': float(ci_spec[0]), 'ci_upper': float(ci_spec[1])},
        'auc': {'point': float(auc_val), 'ci_lower': float(ci_auc[0]), 'ci_upper': float(ci_auc[1])}
    },
    'screening_threshold': {
        'threshold': float(screening_threshold),
        'chosen_on': 'training_set_only',
        'criterion': 'target_training_sensitivity_gte_80',
        'target_sensitivity': TARGET_SENS,
        'test_accuracy': float(acc_screen),
        'test_sensitivity': float(sens_screen),
        'test_specificity': float(spec_screen),
        'test_precision': float(prec_screen),
        'test_f1': float(f1_screen),
        'sensitivity_change_pp': float((sens_screen - sens) * 100),
        'specificity_change_pp': float((spec_screen - spec) * 100),
        'cm': {'tn': int(tn_s), 'fp': int(fp_s), 'fn': int(fn_s), 'tp': int(tp_s)}
    }
}

json_path = os.path.join(FIGURES_DIR, 'extended_analysis_results.json')
with open(json_path, 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"\nResults saved to: {json_path}")

print("\n" + "=" * 60)
print("ALL EXTENDED ANALYSES COMPLETE")
print("=" * 60)
