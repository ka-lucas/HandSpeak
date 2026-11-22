"""Treina e compara classificadores de letras de Libras a partir dos landmarks da mão.

Uso (na raiz do repositório):
    python src/training/train_landmarks_classifier.py

Entrada: data/processed/landmarks.csv (rótulo + 21 pontos x, y, z do MediaPipe).

Avaliação: a base tem muitos quadros repetidos ou quase idênticos (provavelmente extraídos de
vídeos). Numa separação aleatória, quase toda amostra de teste tem uma "gêmea" no treino e a
acurácia fica perto de 100%, o que não reflete o uso real. Por isso:
  1. linhas duplicadas são removidas;
  2. as amostras de cada letra são agrupadas por semelhança de pose (k-means);
  3. a validação cruzada em 5 partes deixa grupos inteiros de fora, então cada parte de teste só
     tem poses que o modelo não viu.

Saídas:
    models/landmarks_classifier.pkl   melhor modelo, treinado com todos os dados
    reports/metrics.md / metrics.json métricas por modelo (média ± desvio nas 5 partes)
    reports/confusion_matrix.png      matriz de confusão do melhor modelo (5 partes somadas)
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "src"))
from utils.normalization import normalize_landmarks_center  # mesma normalização usada na predição

SEED = 42
MIN_SAMPLES_PER_CLASS = 10
POSE_GROUPS_PER_CLASS = 10
N_FOLDS = 5
CSV_PATH = os.path.join(ROOT, "data", "processed", "landmarks.csv")
MODEL_PATH = os.path.join(ROOT, "models", "landmarks_classifier.pkl")
REPORT_DIR = os.path.join(ROOT, "reports")


def make_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=SEED),
    }


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["label"].astype(str).str.strip()
    counts = df["label"].value_counts()
    removed = sorted(counts[counts < MIN_SAMPLES_PER_CLASS].index)
    df = df[df["label"].isin(counts[counts >= MIN_SAMPLES_PER_CLASS].index)]
    n_before = len(df)
    df = df.drop_duplicates()
    n_dup = n_before - len(df)
    X = np.stack([normalize_landmarks_center(row.astype(np.float64).copy())
                  for row in df.drop(columns="label").values])
    return X, df["label"].values, removed, n_dup


def pose_group_folds(X, y):
    """Atribui cada amostra a uma das N_FOLDS partes, mantendo grupos de pose inteiros juntos.

    Feito letra a letra, para todas as letras aparecerem em todas as partes.
    """
    fold = np.empty(len(y), dtype=int)
    rng = np.random.default_rng(SEED)
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        clusters = KMeans(POSE_GROUPS_PER_CLASS, n_init=3, random_state=SEED).fit_predict(X[idx])
        order = rng.permutation(POSE_GROUPS_PER_CLASS)
        for position, c in enumerate(order):
            fold[idx[clusters == c]] = position % N_FOLDS
    return fold


def main():
    X, y, removed, n_dup = load_data()
    labels = sorted(np.unique(y))
    print(f"{len(y)} amostras, {len(labels)} classes ({n_dup} duplicadas removidas; sem amostras suficientes: {removed})")

    fold = pose_group_folds(X, y)
    scores = {name: {"acc": [], "f1": [], "time": []} for name in make_models()}
    oof_pred = {name: np.empty(len(y), dtype=object) for name in make_models()}
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        for name, model in make_models().items():
            t0 = time.time()
            pred = model.fit(X[tr], y[tr]).predict(X[te])
            scores[name]["acc"].append(accuracy_score(y[te], pred))
            scores[name]["f1"].append(f1_score(y[te], pred, average="macro"))
            scores[name]["time"].append(time.time() - t0)
            oof_pred[name][te] = pred
        print(f"parte {k + 1}/{N_FOLDS} ok")

    table = pd.DataFrame({
        name: {"Acurácia": f"{np.mean(s['acc']):.3f} ± {np.std(s['acc']):.3f}",
               "F1 macro": f"{np.mean(s['f1']):.3f} ± {np.std(s['f1']):.3f}",
               "Treino (s)": f"{np.mean(s['time']):.1f}",
               "_acc": np.mean(s["acc"])}
        for name, s in scores.items()}).T
    best_name = table["_acc"].astype(float).idxmax()
    table = table.drop(columns="_acc")
    print(table)

    # Referência: separação aleatória (otimista)
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    random_split_acc = accuracy_score(yb, make_models()[best_name].fit(Xa, ya).predict(Xb))
    print(f"Melhor: {best_name}. Referência com separação aleatória: {random_split_acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    joblib.dump(make_models()[best_name].fit(X, y), MODEL_PATH)

    cm = confusion_matrix(y, oof_pred[best_name], labels=labels)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm_pct[i, j] >= 1:
                ax.text(j, i, f"{cm_pct[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if cm_pct[i, j] > 50 else "black")
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title(f"{best_name}: matriz de confusão (% por letra), validação cruzada por grupos de pose")
    fig.colorbar(im, label="% das amostras da letra"); fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=120)

    report = classification_report(y, oof_pred[best_name], digits=3)
    with open(os.path.join(REPORT_DIR, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("# Métricas do classificador de landmarks\n\n")
        f.write(f"- {len(y)} amostras após remover {n_dup} linhas duplicadas; {len(labels)} letras: {', '.join(labels)}\n")
        f.write(f"- Sem amostras suficientes (fora do estudo): {', '.join(removed)}\n")
        f.write(f"- Validação cruzada em {N_FOLDS} partes por grupos de pose ({POSE_GROUPS_PER_CLASS} grupos k-means por letra): "
                "cada parte de teste só tem poses que o modelo não viu\n\n")
        f.write("## Comparação dos modelos (média ± desvio nas partes)\n\n")
        f.write(table.to_markdown() + "\n\n")
        f.write(f"## Por letra — {best_name}\n\n```\n{report}```\n\n")
        f.write("## Referência: separação aleatória\n\n")
        f.write(f"Com uma separação aleatória comum (80/20), o mesmo modelo chega a {random_split_acc:.3f} de acurácia. "
                "Esse número é otimista: muitas amostras de teste têm uma quase-cópia no treino.\n")
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "random_split_accuracy": random_split_acc,
                   "cv": {n: {k: list(map(float, v)) for k, v in s.items()} for n, s in scores.items()}},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
