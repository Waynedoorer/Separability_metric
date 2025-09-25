from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ============== Load SECOM ==============
df_secom = pd.read_csv("Data/uci-secom/uci-secom.csv")
X = df_secom.iloc[:, 1:-1]
y = df_secom.iloc[:, -1].replace({-1: 0}).astype(int)
TEST_SIZE = 0.2

# ============== Split ==============
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ============== Helper to evaluate a single model ==============
def evaluate_binary_model(model, X_train, y_train, X_test, y_test, name="model"):
    model.fit(X_train, y_train)

    # Get positive-class scores
    if hasattr(model[-1], "predict_proba"):
        p = model.predict_proba(X_test)[:, 1]
    elif hasattr(model[-1], "decision_function"):
        d = model.decision_function(X_test)
        dmin, dmax = d.min(), d.max()
        p = (d - dmin) / (dmax - dmin + 1e-12)
    else:
        p = model.predict(X_test).astype(float)

    yhat = (p >= 0.5).astype(int)

    roc = roc_auc_score(y_test, p)
    pr  = average_precision_score(y_test, p)
    f1  = f1_score(y_test, yhat, zero_division=0)
    acc = accuracy_score(y_test, yhat)

    print(f"\n=== {name} (SECOM, single split) ===")
    print(f"Test samples: {len(y_test)} | Pos: {int(y_test.sum())} | Neg: {int((1-y_test).sum())}")
    print(f"ROC-AUC : {roc:.3f}")
    print(f"PR-AUC  : {pr:.3f}")
    print(f"F1      : {f1:.3f}")
    print(f"Accuracy: {acc:.3f}")

# ============== # classifiers for SECOM: Logistic Regression ==============
logreg_best = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        C=1.0,                 # <-- adjust if you have tuned value
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])

evaluate_binary_model(logreg_best, X_tr, y_tr, X_te, y_te, name="LogisticRegression")


# =========================
# Load SPF and build a classifier for it
# =========================
df = pd.read_csv("Data/steel-plate/faults.csv")
X = df.iloc[:, :27].copy()
fault_cols = list(df.columns[27:])  # 7 binary fault indicators

fault_mat = df[fault_cols].astype(int).values   # (n,7)
fault_sum = fault_mat.sum(axis=1)

# y_raw: 0=no_fault; 1..7=exactly one fault; 8=multi_fault (>=2)
y_raw = np.full(len(df), 8, dtype=int)
y_raw[fault_sum == 0] = 0
single_idx = np.where(fault_sum == 1)[0]
if len(single_idx) > 0:
    single_fault_pos = fault_mat[single_idx].argmax(axis=1)  # 0..6
    y_raw[single_idx] = single_fault_pos + 1                  # 1..7

# Remap to contiguous class ids 0..K-1
classes = np.unique(y_raw)
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.vectorize(class_to_idx.get)(y_raw)
K = len(classes)

# One-hot builder
def to_one_hot(y_int, K):
    M = np.zeros((len(y_int), K), dtype=int)
    M[np.arange(len(y_int)), y_int] = 1
    return M

# =========================
# Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (fit on train, apply to test)
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test  = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

# =========================
# Best params (paste your tuned ones here)
# =========================
best_cls_params = {
    'grow_policy': 'depthwise',
    'n_estimators': 600,
    'learning_rate': 0.05,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'max_depth': 5,
    'min_child_weight': 2,
    'reg_lambda': 1.0,
    'reg_alpha': 0.0,
}

params = dict(best_cls_params)
params.update({
    'objective': 'multi:softprob',
    'num_class': K,
    'tree_method': 'hist',
    'verbosity': 0,
    # 'device': 'cuda',  # enable if your xgboost build supports GPU
})

# =========================
# Train once on train set
# =========================
clf = XGBClassifier(**params, n_jobs=-1)
clf.fit(X_train, y_train)

# =========================
# Predict probabilities on test
# =========================
proba_test = clf.predict_proba(X_test)  # shape (n_test, K)

# =========================
# Evaluate PR-AUC (micro)
# =========================
Y_test_onehot = to_one_hot(y_test, K)

# PR-AUC:
# - micro: aggregates decisions across classes
pr_auc_micro = average_precision_score(Y_test_onehot, proba_test, average='micro')

print("\n=== SPF Multiclass ===")
print(f"PR-AUC  (micro):      {pr_auc_micro:.3f}")