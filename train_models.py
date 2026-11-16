"""
Fresh training pipeline for Chronic-Disease-Prediction.
Run from the repository root:
    python train_models.py

The script downloads the compressed CSV from this GitHub repository,
fits preprocessing on the training split only, trains 9 model families,
and saves artifacts under model_files/.
"""

import os
import json
import urllib.request
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TARGET = "disease_label"
DATA_URL = (
    "https://raw.githubusercontent.com/tamilk07/"
    "Chronic-Disease-Prediction/main/"
    "Cleaned_Clinical_Disease_Dataset_v2.csv.gz"
)
ARTIFACT_DIR = "model_files"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ----------------------- Load dataset -----------------------
DATA_FILE = "Cleaned_Clinical_Disease_Dataset_v2.csv.gz"
if not os.path.exists(DATA_FILE):
    print("Downloading dataset from GitHub...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

df = pd.read_csv(DATA_FILE, compression="gzip")
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found.")

df = df.drop_duplicates().reset_index(drop=True)
X = df.drop(columns=[TARGET])
y_text = df[TARGET].astype(str)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

# Split before fitting preprocessing to avoid data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# Ensure dense float32 arrays for neural models and consistent inference.
if hasattr(X_train_p, "toarray"):
    X_train_p = X_train_p.toarray()
    X_test_p = X_test_p.toarray()
X_train_p = np.asarray(X_train_p, dtype=np.float32)
X_test_p = np.asarray(X_test_p, dtype=np.float32)

joblib.dump(preprocessor, f"{ARTIFACT_DIR}/preprocessor.pkl")
joblib.dump(label_encoder, f"{ARTIFACT_DIR}/label_encoder.pkl")
joblib.dump(list(X.columns), f"{ARTIFACT_DIR}/feature_columns.pkl")
joblib.dump(
    {c: (X_train[c].mode(dropna=True).iloc[0] if X_train[c].notna().any() else "")
     for c in categorical_cols}
    | {c: float(pd.to_numeric(X_train[c], errors="coerce").median() or 0.0)
       for c in numeric_cols},
    f"{ARTIFACT_DIR}/feature_defaults.pkl"
)
joblib.dump(categorical_cols, f"{ARTIFACT_DIR}/categorical_columns.pkl")
joblib.dump(numeric_cols, f"{ARTIFACT_DIR}/numeric_columns.pkl")

n_classes = len(label_encoder.classes_)
results = []

def record_result(name, y_pred):
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred)
    })

models = {}

# ----------------------- Classical ML models -----------------------
print("\nTraining Logistic Regression...")
models["Logistic Regression"] = LogisticRegression(
    max_iter=500, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
)

print("Training Random Forest...")
models["Random Forest"] = RandomForestClassifier(
    n_estimators=250, max_depth=24, min_samples_leaf=2,
    class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
)

try:
    from xgboost import XGBClassifier
    print("Training XGBoost...")
    models["XGBoost"] = XGBClassifier(
        n_estimators=250, max_depth=7, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", tree_method="hist",
        n_jobs=-1, random_state=RANDOM_STATE
    )
except ImportError:
    print("XGBoost unavailable; skipping.")

try:
    from catboost import CatBoostClassifier
    print("Training CatBoost...")
    models["CatBoost"] = CatBoostClassifier(
        iterations=250, depth=7, learning_rate=0.08,
        loss_function="MultiClass", verbose=False,
        random_seed=RANDOM_STATE, thread_count=-1
    )
except ImportError:
    print("CatBoost unavailable; skipping.")

print("Training MLP...")
models["MLP"] = MLPClassifier(
    hidden_layer_sizes=(128, 64), activation="relu",
    batch_size=512, learning_rate_init=0.001,
    max_iter=35, early_stopping=True,
    random_state=RANDOM_STATE
)

for name, model in models.items():
    print(f"\nFitting {name}...")
    model.fit(X_train_p, y_train)
    pred = np.asarray(model.predict(X_test_p)).astype(int).ravel()
    record_result(name, pred)
    joblib.dump(model, f"{ARTIFACT_DIR}/{name.lower().replace(' ', '_')}.pkl")

# ----------------------- PyTorch tabular models -----------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_p.shape[1]

    class FeatureTransformer(nn.Module):
        """Feature-token transformer used for TabTransformer/FT-Transformer/SAINT baselines."""
        def __init__(self, input_dim, n_classes, d_model=32, n_heads=4, n_layers=2, dropout=0.15, mode="ft"):
            super().__init__()
            self.input_dim = input_dim
            self.mode = mode
            self.value_proj = nn.Linear(1, d_model)
            self.feature_embedding = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 2, dropout=dropout,
                batch_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, 64), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(64, n_classes)
            )

        def forward(self, x):
            tokens = self.value_proj(x.unsqueeze(-1)) + self.feature_embedding
            z = self.norm(self.encoder(tokens))
            return self.head(z.mean(dim=1))

    class NODEBaseline(nn.Module):
        """Compact differentiable neural decision-ensemble baseline."""
        def __init__(self, input_dim, n_classes, hidden=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.BatchNorm1d(hidden), nn.Dropout(0.2),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Dropout(0.15), nn.Linear(hidden // 2, n_classes)
            )
        def forward(self, x):
            return self.net(x)

    # Cap training rows for practical CPU deployment/training time.
    max_torch_rows = min(60000, len(X_train_p))
    rng = np.random.default_rng(RANDOM_STATE)
    chosen = rng.choice(len(X_train_p), size=max_torch_rows, replace=False)
    xtr = torch.tensor(X_train_p[chosen], dtype=torch.float32)
    ytr = torch.tensor(y_train[chosen], dtype=torch.long)
    xte = torch.tensor(X_test_p, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(xtr, ytr), batch_size=512, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(xte, yte), batch_size=1024, shuffle=False
    )

    neural_models = {
        "TabTransformer": FeatureTransformer(input_dim, n_classes, mode="tab"),
        "FT-Transformer": FeatureTransformer(input_dim, n_classes, mode="ft"),
        "SAINT": FeatureTransformer(input_dim, n_classes, mode="saint"),
        "NODE": NODEBaseline(input_dim, n_classes)
    }

    for name, net in neural_models.items():
        print(f"\nTraining {name} on {device}...")
        net = net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        epochs = 6
        for epoch in range(epochs):
            net.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"{name} | epoch {epoch + 1}/{epochs} | loss {total_loss / max(1, len(train_loader)):.4f}")

        net.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                logits = net(xb.to(device))
                preds.extend(logits.argmax(dim=1).cpu().numpy())

        record_result(name, np.asarray(preds))
        torch.save({
            "model_name": name,
            "state_dict": net.cpu().state_dict(),
            "input_dim": input_dim,
            "n_classes": n_classes,
            "class_names": label_encoder.classes_.tolist(),
            "architecture": "NODEBaseline" if name == "NODE" else "FeatureTransformer",
            "config": {"d_model": 32, "n_heads": 4, "n_layers": 2, "dropout": 0.15}
        }, f"{ARTIFACT_DIR}/{name.lower().replace('-', '_')}.pth")

except ImportError:
    print("\nPyTorch not installed; neural models skipped.")

# ----------------------- Save results and metadata -----------------------
metrics = pd.DataFrame(results).sort_values("Macro F1", ascending=False)
metrics.to_csv(f"{ARTIFACT_DIR}/model_metrics.csv", index=False)

with open(f"{ARTIFACT_DIR}/training_metadata.json", "w") as f:
    json.dump({
        "target": TARGET,
        "feature_columns": list(X.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "class_names": label_encoder.classes_.tolist(),
        "processed_feature_count": int(input_dim),
        "test_size": 0.20,
        "random_state": RANDOM_STATE
    }, f, indent=2)

print("\n================ TRAINING COMPLETE ================")
print(metrics.to_string(index=False))
print(f"\nArtifacts saved in: {ARTIFACT_DIR}/")
