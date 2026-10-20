import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# Output directory for paper figures
FIGURES_DIR = 'paper_figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------------
# PART 1: DEXA MODEL LEAKAGE CHECK (original.csv)
# -------------------------------------------------------------------
print("=" * 60)
print("PART 1: DEXA MODEL LEAKAGE CHECK (original.csv)")
print("=" * 60)

orig_path = os.path.join('models', 'ML_MODEL(WITH DEXA)', 'original.csv')
orig_df = pd.read_csv(orig_path)

features_20 = [
    'FNT', 'Calcitriol', 'URIC', 'ALT', 'BUN', 'CREA', 'FBG', 'LDL-C', 'L1.4T',
    'Age', 'HDL-C', 'BMI', 'Ca', 'P', 'Height', 'AST', 'Weight', 'Calsium',
    'Calcitonin', 'AS'
]
features_18_no_t = [f for f in features_20 if f not in ['FNT', 'L1.4T']]
TARGET = 'OP'

# Common ExtraTrees parameters from best_model_pipeline.joblib
et_params = {
    'n_estimators': 400,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': False,
    'random_state': 42,
    'n_jobs': -1
}

def evaluate_pipeline(X, y, feature_set_name, balance_method='oversample'):
    # Step 1: Split FIRST (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
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
    if balance_method == 'oversample':
        # Separate majority and minority in train set
        train_combined = pd.concat([X_train_s, y_train], axis=1)
        majority = train_combined[train_combined[TARGET] == 0]
        minority = train_combined[train_combined[TARGET] == 1]
        
        minority_oversampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=42
        )
        train_balanced = pd.concat([majority, minority_oversampled])
        X_tr = train_balanced.drop(columns=[TARGET])
        y_tr = train_balanced[TARGET]
        clf = ExtraTreesClassifier(**et_params)
    elif balance_method == 'class_weight':
        X_tr = X_train_s
        y_tr = y_train
        params = et_params.copy()
        params['class_weight'] = 'balanced'
        clf = ExtraTreesClassifier(**params)
        
    clf.fit(X_tr, y_tr)
    
    # Step 5: Test on UNTOUCHED test set
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
    
    # Step 6: 5-Fold Stratified Cross-Validation (Leakage-Free: Impute, Scale, Balance inside fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    cv_aucs = []
    
    for tr_idx, val_idx in skf.split(X, y):
        X_cv_tr, X_cv_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_cv_tr, y_cv_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        cv_imp = SimpleImputer(strategy='median')
        X_cv_tr_imp = cv_imp.fit_transform(X_cv_tr)
        X_cv_val_imp = cv_imp.transform(X_cv_val)
        
        cv_scl = StandardScaler()
        X_cv_tr_s = cv_scl.fit_transform(X_cv_tr_imp)
        X_cv_val_s = cv_scl.transform(X_cv_val_imp)
        
        if balance_method == 'oversample':
            tr_df = pd.DataFrame(X_cv_tr_s)
            tr_df['y'] = y_cv_tr.values
            maj = tr_df[tr_df['y'] == 0]
            mino = tr_df[tr_df['y'] == 1]
            mino_res = resample(mino, replace=True, n_samples=len(maj), random_state=42)
            bal_df = pd.concat([maj, mino_res])
            X_fold_tr = bal_df.drop(columns=['y'])
            y_fold_tr = bal_df['y']
            fold_clf = ExtraTreesClassifier(**et_params)
        else:
            X_fold_tr = X_cv_tr_s
            y_fold_tr = y_cv_tr
            p = et_params.copy()
            p['class_weight'] = 'balanced'
            fold_clf = ExtraTreesClassifier(**p)
            
        fold_clf.fit(X_fold_tr, y_fold_tr)
        val_pred = fold_clf.predict(X_cv_val_s)
        val_proba = fold_clf.predict_proba(X_cv_val_s)[:, 1]
        
        cv_accs.append(accuracy_score(y_cv_val, val_pred))
        cv_aucs.append(roc_auc_score(y_cv_val, val_proba))
        
    return {
        'name': feature_set_name,
        'model': clf,
        'X_test': X_test_s,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'cm': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'acc': acc, 'prec': prec, 'rec': rec, 'spec': spec, 'f1': f1, 'auc': auc,
        'cv_acc_mean': np.mean(cv_accs), 'cv_acc_std': np.std(cv_accs),
        'cv_auc_mean': np.mean(cv_aucs), 'cv_auc_std': np.std(cv_aucs)
    }

# Run both DEXA versions
results_20 = evaluate_pipeline(orig_df[features_20], orig_df[TARGET], 'ExtraTrees (All 20 Features)')
results_18 = evaluate_pipeline(orig_df[features_18_no_t], orig_df[TARGET], 'ExtraTrees (Without FNT & L1-4T)')

print("\n--- RESULTS TABLE: DEXA MODEL ---")
dexa_table = pd.DataFrame([
    {
        'Model': results_20['name'],
        'Test Acc': f"{results_20['acc']:.4f}",
        'Precision': f"{results_20['prec']:.4f}",
        'Recall (Sens)': f"{results_20['rec']:.4f}",
        'Specificity': f"{results_20['spec']:.4f}",
        'F1 Score': f"{results_20['f1']:.4f}",
        'ROC-AUC': f"{results_20['auc']:.4f}",
        '5-Fold CV Acc': f"{results_20['cv_acc_mean']:.4f} +/- {results_20['cv_acc_std']:.4f}",
        '5-Fold CV AUC': f"{results_20['cv_auc_mean']:.4f} +/- {results_20['cv_auc_std']:.4f}"
    },
    {
        'Model': results_18['name'],
        'Test Acc': f"{results_18['acc']:.4f}",
        'Precision': f"{results_18['prec']:.4f}",
        'Recall (Sens)': f"{results_18['rec']:.4f}",
        'Specificity': f"{results_18['spec']:.4f}",
        'F1 Score': f"{results_18['f1']:.4f}",
        'ROC-AUC': f"{results_18['auc']:.4f}",
        '5-Fold CV Acc': f"{results_18['cv_acc_mean']:.4f} +/- {results_18['cv_acc_std']:.4f}",
        '5-Fold CV AUC': f"{results_18['cv_auc_mean']:.4f} +/- {results_18['cv_auc_std']:.4f}"
    }
])
print(dexa_table.to_string(index=False))

print("\nConfusion Matrix (All 20 Features):")
print(results_20['cm'])
print(f"TN: {results_20['tn']}, FP: {results_20['fp']}, FN: {results_20['fn']}, TP: {results_20['tp']}")

print("\nConfusion Matrix (Without T-scores):")
print(results_18['cm'])
print(f"TN: {results_18['tn']}, FP: {results_18['fp']}, FN: {results_18['fn']}, TP: {results_18['tp']}")

# -------------------------------------------------------------------
# PART 2: WITHOUT-DEXA MODEL (original2.csv)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2: WITHOUT-DEXA MODEL (original2.csv)")
print("=" * 60)

orig2_path = os.path.join('models', 'ML_MODEL(WITHOUTDEXA)', 'original2.csv')
orig2_df = pd.read_csv(orig2_path)

print(f"original2.csv rows: {orig2_df.shape[0]}, columns: {orig2_df.shape[1]}")
print("Class counts for Osteoporosis:\n", orig2_df['Osteoporosis'].value_counts())

# Preprocessing mapping matching train_and_save.py / app.py
def encode_col(val, target_vals):
    return 1 if val in target_vals else 0

df2 = orig2_df.copy()
df2['Gender'] = df2['Gender'].apply(lambda x: encode_col(x, ['Female']))
df2['Hormonal_Changes'] = df2['Hormonal Changes'].apply(lambda x: encode_col(x, ['Postmenopausal']))
df2['Body_Weight'] = df2['Body Weight'].apply(lambda x: encode_col(x, ['Underweight', 'Overweight']))
df2['Vitamin_D_Intake'] = df2['Vitamin D Intake'].apply(lambda x: encode_col(x, ['Sufficient']))
df2['Physical_Activity'] = df2['Physical Activity'].apply(lambda x: encode_col(x, ['Active']))
df2['Smoking'] = df2['Smoking'].apply(lambda x: encode_col(x, ['Yes']))
df2['Alcohol_Consumption'] = df2['Alcohol Consumption'].fillna('None').apply(lambda x: 0 if x == 'None' else 1)
df2['Medications'] = df2['Medications'].fillna('None').apply(lambda x: 0 if str(x).lower() == 'none' else 1)
df2['Prior_Fractures'] = df2['Prior Fractures'].apply(lambda x: encode_col(x, ['Yes']))

nodexa_features = [
    'Age', 'Smoking', 'Physical_Activity', 'Medications', 'Vitamin_D_Intake', 
    'Hormonal_Changes', 'Gender', 'Body_Weight', 'Prior_Fractures', 'Alcohol_Consumption'
]
X2 = df2[nodexa_features]
y2 = df2['Osteoporosis']

# Leakage-safe 70 / 15 / 15 split matching notebook
X2_train, X2_temp, y2_train, y2_temp = train_test_split(
    X2, y2, test_size=0.30, stratify=y2, random_state=42
)
X2_val, X2_test, y2_val, y2_test = train_test_split(
    X2_temp, y2_temp, test_size=0.50, stratify=y2_temp, random_state=42
)

# Impute and scale using train ONLY
scaler2 = StandardScaler()
X2_train_s = pd.DataFrame(scaler2.fit_transform(X2_train), columns=X2.columns, index=X2_train.index)
X2_val_s = pd.DataFrame(scaler2.transform(X2_val), columns=X2.columns, index=X2_val.index)
X2_test_s = pd.DataFrame(scaler2.transform(X2_test), columns=X2.columns, index=X2_test.index)

# Train Gradient Boosting with tuned parameters
gb_params = {
    'n_estimators': 300,
    'max_depth': 2,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'min_samples_leaf': 1,
    'random_state': 42
}
gb_model = GradientBoostingClassifier(**gb_params)
gb_model.fit(X2_train_s, y2_train)

# Test set predictions
y2_pred = gb_model.predict(X2_test_s)
y2_proba = gb_model.predict_proba(X2_test_s)[:, 1]

cm2 = confusion_matrix(y2_test, y2_pred)
tn2, fp2, fn2, tp2 = cm2.ravel()

acc2 = accuracy_score(y2_test, y2_pred)
prec2 = precision_score(y2_test, y2_pred) # Class 1
rec2 = recall_score(y2_test, y2_pred)    # Class 1
spec2 = tn2 / (tn2 + fp2)                # Class 0 TNR
f1_2 = f1_score(y2_test, y2_pred)        # Class 1
auc2 = roc_auc_score(y2_test, y2_proba)

# Macro averages for comparison
macro_prec = precision_score(y2_test, y2_pred, average='macro')
macro_rec = recall_score(y2_test, y2_pred, average='macro')
macro_f1 = f1_score(y2_test, y2_pred, average='macro')

print("\n--- RESULTS: GRADIENT BOOSTING (WITHOUT DEXA) ---")
print(f"Test Accuracy           : {acc2:.4f} ({acc2*100:.2f}%)")
print(f"Class 1 Precision       : {prec2:.4f} ({prec2*100:.2f}%)")
print(f"Class 1 Recall (Sens)   : {rec2:.4f} ({rec2*100:.2f}%)")
print(f"Class 1 F1-Score        : {f1_2:.4f}")
print(f"Specificity (Class 0)   : {spec2:.4f} ({spec2*100:.2f}%)")
print(f"ROC-AUC                 : {auc2:.4f}")
print(f"Macro Avg Precision     : {macro_prec:.4f}")
print(f"Macro Avg Recall        : {macro_rec:.4f}")
print(f"Macro Avg F1            : {macro_f1:.4f}")

print("\nConfusion Matrix (Without DEXA):")
print(cm2)
print(f"TN: {tn2}, FP: {fp2}, FN: {fn2}, TP: {tp2}")

# -------------------------------------------------------------------
# PART 4: SAVE PUBLICATION FIGURES (300 DPI)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("SAVING 300 DPI FIGURES IN paper_figures/")
print("=" * 60)

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# 1. Confusion Matrix - DEXA All 20 Features
fig, ax = plt.subplots(figsize=(5, 4.2), dpi=300)
sns.heatmap(results_20['cm'], annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal (0)', 'Osteoporosis (1)'],
            yticklabels=['Normal (0)', 'Osteoporosis (1)'], ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('ExtraTrees Confusion Matrix\n(All 20 Features, Untouched Test Set)', fontsize=12, pad=12, weight='bold')
ax.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
ax.set_ylabel('True Label', fontsize=11, labelpad=8)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'cm_dexa_all20.png'), dpi=300)
plt.close(fig)

# 2. Confusion Matrix - DEXA Without T-Scores
fig, ax = plt.subplots(figsize=(5, 4.2), dpi=300)
sns.heatmap(results_18['cm'], annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal (0)', 'Osteoporosis (1)'],
            yticklabels=['Normal (0)', 'Osteoporosis (1)'], ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('ExtraTrees Confusion Matrix\n(Without T-Scores, Untouched Test Set)', fontsize=12, pad=12, weight='bold')
ax.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
ax.set_ylabel('True Label', fontsize=11, labelpad=8)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'cm_dexa_no_tscores.png'), dpi=300)
plt.close(fig)

# 3. Confusion Matrix - Without-DEXA Gradient Boosting
fig, ax = plt.subplots(figsize=(5, 4.2), dpi=300)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Normal (0)', 'Osteoporosis (1)'],
            yticklabels=['Normal (0)', 'Osteoporosis (1)'], ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('Gradient Boosting Confusion Matrix\n(Clinical Triage / Without DEXA)', fontsize=12, pad=12, weight='bold')
ax.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
ax.set_ylabel('True Label', fontsize=11, labelpad=8)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'cm_without_dexa_gb.png'), dpi=300)
plt.close(fig)

# 4. ROC Curves
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

fpr20, tpr20, _ = roc_curve(results_20['y_test'], results_20['y_proba'])
fpr18, tpr18, _ = roc_curve(results_18['y_test'], results_18['y_proba'])
fpr_gb, tpr_gb, _ = roc_curve(y2_test, y2_proba)

ax.plot(fpr20, tpr20, color='#1f77b4', lw=2.2, label=f'DEXA ExtraTrees (All 20) (AUC = {results_20["auc"]:.3f})')
ax.plot(fpr18, tpr18, color='#ff7f0e', lw=2.2, label=f'DEXA ExtraTrees (No T-scores) (AUC = {results_18["auc"]:.3f})')
ax.plot(fpr_gb, tpr_gb, color='#2ca02c', lw=2.2, label=f'Clinical Gradient Boosting (AUC = {auc2:.3f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--', label='Random Guess')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
ax.set_title('ROC Curves Comparison (Untouched Test Sets)', fontsize=12, weight='bold', pad=12)
ax.legend(loc='lower right', frameon=True, fontsize=9.5)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'roc_curves_combined.png'), dpi=300)
plt.close(fig)

# Individual ROC curves
for fname, fpr, tpr, auc_val, title, color in [
    ('roc_dexa_all20.png', fpr20, tpr20, results_20['auc'], 'ROC Curve - ExtraTrees (All 20 Features)', '#1f77b4'),
    ('roc_dexa_no_tscores.png', fpr18, tpr18, results_18['auc'], 'ROC Curve - ExtraTrees (Without T-Scores)', '#ff7f0e'),
    ('roc_without_dexa_gb.png', fpr_gb, tpr_gb, auc2, 'ROC Curve - Gradient Boosting (Without DEXA)', '#2ca02c')
]:
    f, a = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    a.plot(fpr, tpr, color=color, lw=2.2, label=f'ROC (AUC = {auc_val:.4f})')
    a.plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--')
    a.set_xlim([-0.02, 1.02])
    a.set_ylim([-0.02, 1.02])
    a.set_xlabel('False Positive Rate', fontsize=11)
    a.set_ylabel('True Positive Rate', fontsize=11)
    a.set_title(title, fontsize=11, weight='bold', pad=10)
    a.legend(loc='lower right', frameon=True)
    a.grid(alpha=0.3)
    plt.tight_layout()
    f.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)
    plt.close(f)

# 5. Feature Importance Chart (Top 10 Features)
fi_series = pd.Series(results_20['model'].feature_importances_, index=features_20).sort_values(ascending=False)
top10_fi = fi_series.head(10)

fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
sns.barplot(x=top10_fi.values, y=top10_fi.index, palette='viridis', ax=ax)
ax.set_title('Top 10 Feature Importances — ExtraTrees (DEXA Model)', fontsize=12, weight='bold', pad=12)
ax.set_xlabel('Relative Importance (Gini)', fontsize=11)
ax.set_ylabel('Feature Name', fontsize=11)
for i, v in enumerate(top10_fi.values):
    ax.text(v + 0.003, i, f"{v:.4f}", va='center', fontsize=9.5, weight='bold')
ax.set_xlim(0, max(top10_fi.values) * 1.15)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'feature_importance_top10.png'), dpi=300)
plt.close(fig)

print("All figures successfully generated and saved to paper_figures/!")
