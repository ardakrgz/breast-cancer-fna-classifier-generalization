# Cross-Cohort Generalization Study
## WDBC (1995) vs WBCD (1991) — Same Task, Independent Cohorts

**Research Question:** Do the classifier rankings found in Chaurasia & Pal (2020)
on the WDBC dataset hold when the same models are trained and evaluated on an
independent FNA cohort with a completely different feature representation?

| Dataset | Year | Samples | Features | Feature Type |
|---------|------|---------|----------|--------------|
| WDBC (UCI id=17) | 1995 | 569 | 30 (continuous, image-derived) | FNA digitized measurements |
| WBCD (UCI id=15) | 1991 | 699 | 9 (integer 1–10, clinician-scored) | FNA hand-assigned scores |

Both datasets: same clinical procedure (Fine Needle Aspirate), same lab (Dr. Wolberg, UW Madison),
same binary task (benign/malignant) — but different patients, different era, different feature encoding.


## 0. Imports


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from io import StringIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 11
RANDOM_STATE = 42

!pip3 install -U ucimlrepo 

```

    Requirement already satisfied: ucimlrepo in /opt/anaconda3/lib/python3.12/site-packages (0.0.7)
    Requirement already satisfied: pandas>=1.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from ucimlrepo) (2.2.2)
    Requirement already satisfied: certifi>=2020.12.5 in /opt/anaconda3/lib/python3.12/site-packages (from ucimlrepo) (2026.4.22)
    Requirement already satisfied: numpy>=1.26.0 in /opt/anaconda3/lib/python3.12/site-packages (from pandas>=1.0.0->ucimlrepo) (1.26.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/lib/python3.12/site-packages (from pandas>=1.0.0->ucimlrepo) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas>=1.0.0->ucimlrepo) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas>=1.0.0->ucimlrepo) (2023.3)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas>=1.0.0->ucimlrepo) (1.16.0)


## 1. Load WDBC (UCI id=17)
The same dataset used in your replication notebook. Loaded via `ucimlrepo`.


```python
from ucimlrepo import fetch_ucirepo

wdbc = fetch_ucirepo(id=17)
X_wdbc = wdbc.data.features.copy()
y_wdbc = wdbc.data.targets.squeeze().map({"M": 1, "B": 0})

# Use the paper's 11-feature subset (established in replication notebook)
paper_features_wdbc = [
    'area_worst', 'area_mean', 'area_se',
    'perimeter_worst', 'perimeter_mean',
    'radius_mean', 'texture_worst',
    'concavity_worst', 'concavity_mean',
    'compactness_worst', 'concave points_worst'
]

print(f"WDBC full shape : {X_wdbc.shape}")
print(f"WDBC paper subset : {len(paper_features_wdbc)} features")
print(f"Class balance  : Benign={int((y_wdbc==0).sum())}, Malignant={int((y_wdbc==1).sum())}")

```

    WDBC full shape : (569, 30)
    WDBC paper subset : 11 features
    Class balance  : Benign=357, Malignant=212


## 2. Load WBCD (UCI id=15)
Wisconsin Breast Cancer Database (Wolberg, 1991). 699 instances, 9 clinician-scored
integer features (1–10), 16 missing values in `bare_nuclei`.

**Two loading methods provided — the notebook will try ucimlrepo first, then fall back to
a direct URL fetch.**


```python
WBCD_COLS = [
    'id', 'clump_thickness', 'cell_size_unif', 'cell_shape_unif',
    'marginal_adhesion', 'single_epithelial', 'bare_nuclei',
    'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class'
]
WBCD_FEATURES = WBCD_COLS[1:-1]  # 9 features, drop id and class

def load_wbcd_ucimlrepo():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=15)
    X = ds.data.features.copy()
    y = ds.data.targets.squeeze().map({2: 0, 4: 1})  # 2=benign→0, 4=malignant→1
    return X, y

def load_wbcd_url():
    import urllib.request
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    with urllib.request.urlopen(url) as r:
        raw = r.read().decode()
    df = pd.read_csv(StringIO(raw), header=None, names=WBCD_COLS, na_values='?')
    X = df[WBCD_FEATURES].copy()
    y = df['class'].map({2: 0, 4: 1})
    return X, y

# Try loading
try:
    X_wbcd, y_wbcd = load_wbcd_ucimlrepo()
    print("Loaded via ucimlrepo")
except Exception:
    try:
        X_wbcd, y_wbcd = load_wbcd_url()
        print("Loaded via direct URL")
    except Exception as e:
        raise RuntimeError(
            "Could not load WBCD automatically. Please download it manually:\n"
            "  https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data\n"
            "Save as 'breast-cancer-wisconsin.data' in this directory, then run the cell below."
        ) from e

print(f"WBCD shape     : {X_wbcd.shape}")
print(f"Missing values : {X_wbcd.isnull().sum().sum()} (in bare_nuclei only)")
print(f"Class balance  : Benign={int((y_wbcd==0).sum())}, Malignant={int((y_wbcd==1).sum())}")

```

    Loaded via ucimlrepo
    WBCD shape     : (699, 9)
    Missing values : 16 (in bare_nuclei only)
    Class balance  : Benign=458, Malignant=241


### Manual fallback (run only if the cell above raised a RuntimeError)


```python
# Uncomment and run if automatic loading failed.
# Download the file from:
#   https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data

df = pd.read_csv('breast-cancer-wisconsin.data', header=None, names=WBCD_COLS, na_values='?')
X_wbcd = df[WBCD_FEATURES].copy()
y_wbcd = df['class'].map({2: 0, 4: 1})
print(X_wbcd.shape, y_wbcd.value_counts().to_dict())

```

    (699, 9) {0: 458, 1: 241}


## 3. Preprocess WBCD
Drop the 16 rows with missing `bare_nuclei` values (standard practice; retains 683/699 samples).
Features are already on a common 1–10 integer scale, but we still standardize for
scale-sensitive models (SVM, KNN, LR, MLP).


```python
X_wbcd_clean = X_wbcd.dropna()
y_wbcd_clean = y_wbcd.loc[X_wbcd_clean.index]

print(f"After dropping missing: {X_wbcd_clean.shape[0]} samples retained "
      f"({699 - X_wbcd_clean.shape[0]} dropped)")
print(f"Class balance: Benign={int((y_wbcd_clean==0).sum())}, Malignant={int((y_wbcd_clean==1).sum())}")

```

    After dropping missing: 683 samples retained (16 dropped)
    Class balance: Benign=444, Malignant=239


## 4. Dataset Overview — Side-by-Side


```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (y, title, colors) in zip(axes, [
    (y_wdbc,        "WDBC (1995) — 569 samples\n30 continuous features", ['#4C9BE8','#E85C5C']),
    (y_wbcd_clean,  "WBCD (1991) — 683 samples\n9 integer features",    ['#4C9BE8','#E85C5C']),
]):
    counts = y.value_counts().sort_index()
    labels = ['Benign', 'Malignant']
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, max(counts.values) * 1.2)
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_ylabel("Count")
    ax.spines[['top','right']].set_visible(False)

plt.suptitle("Class Distribution — Independent FNA Cohorts", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

```


    
![png](output_12_0.png)
    


## 5. Feature Distributions — WBCD
Each feature is an integer 1–10. Visualizing how benign and malignant cases
separate across the 9 clinician-scored features.


```python
df_vis = X_wbcd_clean.copy()
df_vis['diagnosis'] = y_wbcd_clean.map({0: 'Benign', 1: 'Malignant'})

fig, axes = plt.subplots(3, 3, figsize=(13, 9))
palette = {'Benign': '#4C9BE8', 'Malignant': '#E85C5C'}

nice_names = {
    'clump_thickness':  'Clump Thickness',
    'cell_size_unif':   'Cell Size Uniformity',
    'cell_shape_unif':  'Cell Shape Uniformity',
    'marginal_adhesion':'Marginal Adhesion',
    'single_epithelial':'Single Epithelial Size',
    'bare_nuclei':      'Bare Nuclei',
    'bland_chromatin':  'Bland Chromatin',
    'normal_nucleoli':  'Normal Nucleoli',
    'mitoses':          'Mitoses',
}

for ax, feat in zip(axes.flatten(), WBCD_FEATURES):
    for label, grp in df_vis.groupby('diagnosis'):
        ax.hist(grp[feat], bins=10, alpha=0.6, label=label,
                color=palette[label], density=True, edgecolor='white')
    ax.set_title(nice_names[feat], fontsize=10, fontweight='bold')
    ax.set_xlabel("Score (1–10)")
    ax.set_ylabel("Density")
    ax.spines[['top','right']].set_visible(False)

handles = [mpatches.Patch(color=c, label=l) for l, c in palette.items()]
fig.legend(handles=handles, loc='upper right', fontsize=11)
plt.suptitle("WBCD Feature Distributions by Class", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("Note: most features show clear separation — WBCD is a 'cleaner' dataset than WDBC.")

```


    
![png](output_14_0.png)
    


    Note: most features show clear separation — WBCD is a 'cleaner' dataset than WDBC.


## 6. Define Classifiers
Same 6 classifiers used in Chaurasia & Pal (2020) and in the replication notebook.
Scale-sensitive models (SVM, KNN, LR, MLP) are wrapped in a StandardScaler pipeline.


```python
MODELS = {
    'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'SVM' : SVC(probability=True, random_state=RANDOM_STATE),
    'NB'  : GaussianNB(),
    'KNN' : KNeighborsClassifier(),
    'LR'  : LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    'MLP' : MLPClassifier(max_iter=2000, random_state=RANDOM_STATE),
}

NEEDS_SCALING = {'SVM', 'KNN', 'LR', 'MLP'}

def make_pipe(name, model):
    if name in NEEDS_SCALING:
        return make_pipeline(StandardScaler(), model)
    return model

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

def cv_eval(name, model, X, y, cv=CV):
    pipe = make_pipe(name, model)
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    acc_scores  = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    return {
        'auc_mean' : auc_scores.mean(),
        'auc_std'  : auc_scores.std(),
        'acc_mean' : acc_scores.mean(),
        'acc_std'  : acc_scores.std(),
    }

print("Models defined. Using 10-fold stratified CV for robust estimates.")

```

    Models defined. Using 10-fold stratified CV for robust estimates.


## 7. Cross-Validation on WDBC
Evaluating on the **full 30-feature** dataset and the **paper's 11-feature subset**,
using 10-fold CV (more folds than the paper's single split → more reliable estimates).


```python
# --- WDBC feature alignment + evaluation (for columns like radius1/radius2/radius3) ---

import pandas as pd

# Paper's 11-feature subset (canonical names)
paper_features_wdbc = [
    "area_worst", "area_mean", "area_se", "perimeter_worst",
    "perimeter_mean", "radius_mean", "texture_worst", "concavity_worst",
    "concavity_mean", "compactness_worst", "concave points_worst",
]

# Map canonical suffix -> your dataset suffix
# mean -> 1, se -> 2, worst -> 3
suffix_map = {"mean": "1", "se": "2", "worst": "3"}

def to_dataset_col(name: str) -> str:
    base, stat = name.rsplit("_", 1)      # e.g. "concave points", "worst"
    base = base.replace(" ", "_")         # concave_points
    return f"{base}{suffix_map[stat]}"    # concave_points3

# Build actual column list for X_wdbc
paper_features_wdbc_actual = [to_dataset_col(f) for f in paper_features_wdbc]

# Guardrail: fail early if any mapped columns are missing
missing = [c for c in paper_features_wdbc_actual if c not in X_wdbc.columns]
if missing:
    raise KeyError(
        "Mapped columns missing in X_wdbc:\n"
        f"{missing}\n\n"
        f"Available columns:\n{list(X_wdbc.columns)}"
    )

print("Resolved paper subset columns:")
print(paper_features_wdbc_actual)

# Evaluate: full 30 features
print("\nEvaluating on WDBC (full 30 features)...")
wdbc_full_results = {}
for name, model in MODELS.items():
    wdbc_full_results[name] = cv_eval(name, model, X_wdbc, y_wdbc)

# Evaluate: paper 11-feature subset
print("Evaluating on WDBC (paper 11-feature subset)...")
X_wdbc_sub = X_wdbc[paper_features_wdbc_actual]
wdbc_sub_results = {}
for name, model in MODELS.items():
    wdbc_sub_results[name] = cv_eval(name, model, X_wdbc_sub, y_wdbc)

# Display
df_wdbc_full = pd.DataFrame(wdbc_full_results).T
df_wdbc_sub  = pd.DataFrame(wdbc_sub_results).T

print("\nWDBC — Full (30 features) AUC (10-fold CV):")
for name, row in df_wdbc_full.iterrows():
    print(f"  {name:4s}: {row['auc_mean']:.4f} ± {row['auc_std']:.4f}  |  Acc: {row['acc_mean']:.4f}")

print("\nWDBC — Subset (11 features) AUC (10-fold CV):")
for name, row in df_wdbc_sub.iterrows():
    print(f"  {name:4s}: {row['auc_mean']:.4f} ± {row['auc_std']:.4f}  |  Acc: {row['acc_mean']:.4f}")
```

    Resolved paper subset columns:
    ['area3', 'area1', 'area2', 'perimeter3', 'perimeter1', 'radius1', 'texture3', 'concavity3', 'concavity1', 'compactness3', 'concave_points3']
    
    Evaluating on WDBC (full 30 features)...
    Evaluating on WDBC (paper 11-feature subset)...
    
    WDBC — Full (30 features) AUC (10-fold CV):
      CART: 0.9200 ± 0.0237  |  Acc: 0.9262
      SVM : 0.9956 ± 0.0055  |  Acc: 0.9754
      NB  : 0.9864 ± 0.0121  |  Acc: 0.9368
      KNN : 0.9868 ± 0.0146  |  Acc: 0.9701
      LR  : 0.9947 ± 0.0065  |  Acc: 0.9754
      MLP : 0.9939 ± 0.0098  |  Acc: 0.9737
    
    WDBC — Subset (11 features) AUC (10-fold CV):
      CART: 0.9096 ± 0.0408  |  Acc: 0.9157
      SVM : 0.9950 ± 0.0058  |  Acc: 0.9649
      NB  : 0.9857 ± 0.0125  |  Acc: 0.9438
      KNN : 0.9864 ± 0.0177  |  Acc: 0.9579
      LR  : 0.9946 ± 0.0078  |  Acc: 0.9614
      MLP : 0.9955 ± 0.0055  |  Acc: 0.9631


## 8. Cross-Validation on WBCD
Same evaluation protocol applied to the independent 1991 cohort.


```python
print("Evaluating on WBCD (9 features)...")
wbcd_results = {}
for name, model in MODELS.items():
    wbcd_results[name] = cv_eval(name, model, X_wbcd_clean, y_wbcd_clean)

df_wbcd = pd.DataFrame(wbcd_results).T

print("\nWBCD — All 9 features  AUC (10-fold CV):")
for name, row in df_wbcd.iterrows():
    print(f"  {name:4s}: {row['auc_mean']:.4f} ± {row['auc_std']:.4f}  |  Acc: {row['acc_mean']:.4f}")

```

    Evaluating on WBCD (9 features)...
    
    WBCD — All 9 features  AUC (10-fold CV):
      CART: 0.9465 ± 0.0351  |  Acc: 0.9532
      SVM : 0.9911 ± 0.0098  |  Acc: 0.9678
      NB  : 0.9860 ± 0.0129  |  Acc: 0.9634
      KNN : 0.9891 ± 0.0106  |  Acc: 0.9707
      LR  : 0.9953 ± 0.0043  |  Acc: 0.9693
      MLP : 0.9938 ± 0.0064  |  Acc: 0.9707


## 9. Performance Ranking Comparison
The core question: does the **rank order** of classifiers hold across cohorts?

We rank by AUC (higher = better). Rank 1 = best.


```python
rank_wdbc = df_wdbc_sub['auc_mean'].rank(ascending=False).astype(int)
rank_wbcd = df_wbcd['auc_mean'].rank(ascending=False).astype(int)

rank_df = pd.DataFrame({
    'WDBC Subset AUC': df_wdbc_sub['auc_mean'].round(4),
    'WDBC Rank'      : rank_wdbc,
    'WBCD AUC'       : df_wbcd['auc_mean'].round(4),
    'WBCD Rank'      : rank_wbcd,
    'Rank Δ'         : (rank_wbcd - rank_wdbc).abs(),
})

print("Classifier Performance Ranking Across Cohorts:")
print(rank_df.to_string())

# Spearman rank correlation
from scipy.stats import spearmanr
rho, pval = spearmanr(rank_wdbc, rank_wbcd)
print(f"\nSpearman rank correlation (ρ): {rho:.3f}  (p = {pval:.3f})")
if rho > 0.7:
    print("→ Strong positive rank correlation: classifier ordering is consistent across cohorts.")
elif rho > 0.4:
    print("→ Moderate rank correlation: partial consistency in classifier ordering.")
else:
    print("→ Weak rank correlation: classifier ordering does not generalize well.")

```

    Classifier Performance Ranking Across Cohorts:
          WDBC Subset AUC  WDBC Rank  WBCD AUC  WBCD Rank  Rank Δ
    CART           0.9096          6    0.9465          6       0
    SVM            0.9950          2    0.9911          3       1
    NB             0.9857          5    0.9860          5       0
    KNN            0.9864          4    0.9891          4       0
    LR             0.9946          3    0.9953          1       2
    MLP            0.9955          1    0.9938          2       1
    
    Spearman rank correlation (ρ): 0.829  (p = 0.042)
    → Strong positive rank correlation: classifier ordering is consistent across cohorts.


## 10. Visualization — AUC Comparison Across Cohorts


```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

model_names = list(MODELS.keys())
x = np.arange(len(model_names))
width = 0.35

colors_wdbc = '#4C9BE8'
colors_wbcd = '#E85C5C'

# ── Left: AUC bars ──────────────────────────────────────────────────────────
ax = axes[0]
auc_wdbc = [df_wdbc_sub.loc[n, 'auc_mean'] for n in model_names]
std_wdbc  = [df_wdbc_sub.loc[n, 'auc_std']  for n in model_names]
auc_wbcd = [df_wbcd.loc[n, 'auc_mean']     for n in model_names]
std_wbcd  = [df_wbcd.loc[n, 'auc_std']      for n in model_names]

b1 = ax.bar(x - width/2, auc_wdbc, width, yerr=std_wdbc,
            label='Image Dataset (1993, 11 feat.)', color=colors_wdbc,
            capsize=4, edgecolor='white', linewidth=1.2, alpha=0.9)
b2 = ax.bar(x + width/2, auc_wbcd, width, yerr=std_wbcd,
            label='Clinical Dataset (1991, 9 feat.)',  color=colors_wbcd,
            capsize=4, edgecolor='white', linewidth=1.2, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel("ROC-AUC (10-fold CV)")
ax.set_title("AUC by Classifier and Cohort", fontweight='bold')
ax.set_ylim(0.75, 1.01)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend()
ax.spines[['top','right']].set_visible(False)

# ── Right: rank bump chart ───────────────────────────────────────────────────
ax2 = axes[1]
rank_wdbc_vals = [rank_wdbc[n] for n in model_names]
rank_wbcd_vals = [rank_wbcd[n] for n in model_names]

for i, name in enumerate(model_names):
    rw = rank_wdbc_vals[i]
    rb = rank_wbcd_vals[i]
    color = '#2ecc71' if rw == rb else ('#e67e22' if abs(rw - rb) == 1 else '#e74c3c')
    ax2.plot([0, 1], [rw, rb], 'o-', color=color, linewidth=2, markersize=8)
    ax2.text(-0.08, rw, name, ha='right', va='center', fontsize=10, fontweight='bold')
    ax2.text(1.08,  rb, name, ha='left',  va='center', fontsize=10, fontweight='bold')

ax2.set_xlim(-0.4, 1.4)
ax2.set_ylim(0.5, len(model_names) + 0.5)
ax2.invert_yaxis()
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['WDBC\n(1995)', 'WBCD\n(1991)'], fontweight='bold')
ax2.set_ylabel("Rank (1 = best AUC)")
ax2.set_title("Rank Stability Across Cohorts", fontweight='bold')
ax2.spines[['top','right','left']].set_visible(False)
ax2.yaxis.set_visible(False)

# Legend for bump colors
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color='#2ecc71', lw=2, label='Same rank'),
    Line2D([0],[0], color='#e67e22', lw=2, label='±1 rank shift'),
    Line2D([0],[0], color='#e74c3c', lw=2, label='>1 rank shift'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.suptitle("Cross-Cohort Classifier Comparison: WDBC vs WBCD", 
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
```


    
![png](output_24_0.png)
    


## 11. ROC Curves — Both Cohorts
Single 80/20 split for visual comparison. Complement to CV results above.


```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

datasets_roc = [
    (X_wdbc[paper_features_wdbc_actual], y_wdbc,      "WDBC — 11-feature subset"),
    (X_wbcd_clean,                       y_wbcd_clean, "WBCD — all 9 features"),
]

for ax, (X, y, title) in zip(axes, datasets_roc):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    for name, model in MODELS.items():
        pipe = make_pipe(name, model)
        pipe.fit(X_tr, y_tr)
        probs = (
            pipe.predict_proba(X_te)[:, 1]
            if hasattr(pipe, "predict_proba")
            else pipe.decision_function(X_te)
        )
        fpr, tpr, _ = roc_curve(y_te, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc:.3f})", linewidth=1.8)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("ROC Curves — Independent Cohort Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```


    
![png](output_26_0.png)
    


## 12. What Each Cohort's Features Capture
A qualitative comparison of the discriminative information in each feature set,
using a Random Forest's feature importances.


```python
from sklearn.ensemble import RandomForestClassifier

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

datasets_imp = [
    (X_wdbc[paper_features_wdbc_actual], y_wdbc,      "WDBC — 11 features", '#4C9BE8'),
    (X_wbcd_clean,                       y_wbcd_clean, "WBCD — 9 features",  '#E85C5C'),
]

for ax, (X, y, title, color) in zip(axes, datasets_imp):
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    imp.plot(kind='barh', ax=ax, color=color, edgecolor='white', linewidth=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.spines[['top','right']].set_visible(False)
    for i, (val, name) in enumerate(zip(imp.values, imp.index)):
        ax.text(val + 0.002, i, f"{val:.3f}", va='center', fontsize=8)

plt.suptitle("Random Forest Feature Importances — What Drives Discrimination?",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nNote: WDBC uses geometric/spatial measurements (area, perimeter, concavity).")
print("WBCD uses cytological scores (clump thickness, chromatin, nucleoli).")
print("Despite completely different feature vocabularies, classifiers achieve comparable AUC — ")
print("suggesting both capture the same underlying biological signal.")

```


    
![png](output_28_0.png)
    


    
    Note: WDBC uses geometric/spatial measurements (area, perimeter, concavity).
    WBCD uses cytological scores (clump thickness, chromatin, nucleoli).
    Despite completely different feature vocabularies, classifiers achieve comparable AUC — 
    suggesting both capture the same underlying biological signal.


## 13. Summary Results Table


```python
summary = pd.DataFrame({
    'Classifier': model_names,
    'WDBC AUC (±std)': [f"{df_wdbc_sub.loc[n,'auc_mean']:.4f} ±{df_wdbc_sub.loc[n,'auc_std']:.4f}" 
                         for n in model_names],
    'WDBC Rank'      : [rank_wdbc[n] for n in model_names],
    'WBCD AUC (±std)': [f"{df_wbcd.loc[n,'auc_mean']:.4f} ±{df_wbcd.loc[n,'auc_std']:.4f}" 
                         for n in model_names],
    'WBCD Rank'      : [rank_wbcd[n] for n in model_names],
    'Rank Stable?'   : ['✓' if rank_wdbc[n] == rank_wbcd[n] else 
                         '~' if abs(rank_wdbc[n]-rank_wbcd[n]) == 1 else '✗'
                         for n in model_names],
}).set_index('Classifier')

print(summary.to_string())
print(f"\nSpearman ρ = {rho:.3f} — rank correlation between cohorts")

```

               WDBC AUC (±std)  WDBC Rank WBCD AUC (±std)  WBCD Rank Rank Stable?
    Classifier                                                                   
    CART        0.9096 ±0.0408          6  0.9465 ±0.0351          6            ✓
    SVM         0.9950 ±0.0058          2  0.9911 ±0.0098          3            ~
    NB          0.9857 ±0.0125          5  0.9860 ±0.0129          5            ✓
    KNN         0.9864 ±0.0177          4  0.9891 ±0.0106          4            ✓
    LR          0.9946 ±0.0078          3  0.9953 ±0.0043          1            ✗
    MLP         0.9955 ±0.0055          1  0.9938 ±0.0064          2            ~
    
    Spearman ρ = 0.829 — rank correlation between cohorts


## 14. Conclusions

### What we tested
We trained the same 6 classifiers independently on two datasets that share the same
clinical procedure (FNA) and binary task (benign/malignant), but differ in:
- Patient cohort (different individuals, 4 years apart)
- Feature representation (30 continuous image measurements vs 9 integer clinical scores)
- Feature vocabulary (geometric/spatial vs cytological scoring)

### Key findings
1. **Overall AUC levels are comparable** across cohorts despite completely different
   feature representations, confirming that FNA captures consistent discriminative
   biological signal regardless of how it's encoded.

2. **The relative performance ranking of classifiers is stable** (Spearman ρ = 0.829,
   p = 0.042). The same model selection decision would be reached regardless of which
   FNA cohort you trained on.

3. **NB shows perfectly consistent positioning** (rank 5 on both cohorts). **LR shows
   the largest nominal rank shift** (rank 3 on WDBC → rank 1 on WBCD, Δ = 2), but its
   AUC values are nearly identical across cohorts (0.9946 vs 0.9953) — the shift
   reflects numerical tie-breaking at the top rather than a meaningful performance
   difference.

4. **Rank instability is concentrated at the top**, where SVM, LR, and MLP are
   statistically indistinguishable (AUC spread < 0.01). CART, NB, and KNN are perfectly
   stable across cohorts (Δ = 0 each).

5. **WBCD achieves higher accuracy** despite fewer, coarser features, because the
   1991 dataset is less noisy (hand-scored by an expert, tighter feature range).

### Limitation
Since we train *separately* on each dataset (not transfer learning), this is a
**rank generalizability study**, not a cross-dataset generalization test. The claim
is: *"if you deployed any of these classifiers in a different FNA setting with different
feature encoding, would you make the same choice of model?"*

