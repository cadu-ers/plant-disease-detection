"""
Classificador híbrido e estudo comparativo: treina 3 SVMs sobre as features de
extract_features.py e compara com a CNN no conjunto de TESTE.

Modelos (todos SVC kernel RBF com class_weight="balanced", em Pipeline/ColumnTransformer):
    SVM_classicas : features clássicas -> StandardScaler -> SVC
    SVM_cnn       : embeddings da CNN  -> StandardScaler -> PCA -> SVC
    SVM_hibrido   : [embeddings -> StandardScaler -> PCA ; clássicas -> StandardScaler] -> SVC
    CNN           : softmax da própria CNN (referência)

Decisões de projeto:
    - SVM com kernel RBF: funciona bem com poucas centenas de features densas e
      padronizadas, separa classes de forma não linear e, com class_weight="balanced",
      compensa o desbalanceamento (~21x) entre as classes.
    - PCA no bloco CNN: os embeddings têm 1280 dimensões e as features clássicas
      147. Sem redução, a distância do kernel RBF seria dominada pelo bloco CNN e as
      clássicas quase não pesariam. A PCA (64/128/256 componentes) equilibra os dois
      blocos e remove dimensões redundantes.
    - Hiperparâmetros escolhidos na VALIDAÇÃO, não com CV no treino: a CNN foi
      treinada nessas mesmas imagens de treino, então os embeddings do treino são
      "fáceis demais" (separados demais). Uma CV dentro do treino superestimaria
      todos os candidatos e enviesaria a escolha para modelos que só funcionam nesses
      embeddings. Por isso o GridSearchCV usa um PredefinedSplit: ajusta no treino e
      avalia na validação (imagens que a CNN só usou para early stopping). Com
      refit=True, o modelo final é reajustado em treino+validação com os melhores
      hiperparâmetros. O TESTE nunca entra no ajuste nem na escolha.
    - Teste de McNemar: compara dois classificadores nas MESMAS imagens de teste,
      olhando só os casos em que eles discordam (um acerta e o outro erra). Se as
      diferenças fossem acaso, esses casos se dividiriam 50/50; o teste binomial exato
      dá o p-valor (p < 0.05 = diferença significativa).
    - Controle "refit só no treino" (--control-train-only): no protocolo principal, o
      refit final dos SVMs usa treino+validação (15.438 imagens), enquanto a CNN
      aprendeu seus pesos só no treino (12.714; a validação serviu apenas para o
      early stopping). A comparação CNN x SVMs misturaria, então, o efeito da fusão
      com o efeito de ~21% a mais de dados. O controle reusa os melhores
      hiperparâmetros já escolhidos (lidos de metrics_<modelo>.json, sem refazer a
      grade), ajusta os 3 SVMs SÓ no treino e repete a avaliação no teste. Se a
      vantagem dos SVMs sobre a CNN se mantiver, ela vem da fusão/classificador e não
      dos dados extras.

Uso:
    python src/hybrid_classifier.py                       -> grade completa (protocolo principal)
    python src/hybrid_classifier.py --quick               -> grade reduzida (teste rápido)
    python src/hybrid_classifier.py --control-train-only  -> controle: SVMs ajustados só no treino
    python src/hybrid_classifier.py --replot              -> só regera o comparativo.png principal
    (requer features_cache/ gerado por src/extract_features.py)

Gera (em reports/hybrid/):
    metrics_<modelo>.json, classification_report_<modelo>.txt  (os 4 modelos)
    grid_search_<modelo>.csv                                   (os 3 SVMs)
    confusion_matrix_SVM_hibrido.png
    comparativo.csv, comparativo.png
    mcnemar.txt
e models/svm_hybrid.joblib (melhor pipeline híbrido).
O controle gera metrics_<modelo>.json, classification_report_<modelo>.txt,
comparativo.csv, comparativo.png e mcnemar.txt em reports/hybrid/controle_refit_treino/,
sem tocar nos arquivos principais nem no modelo salvo.
"""

import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from config import (  # noqa: E402
    CLASS_NAMES_PATH, CNN_EMBEDDINGS_PATH, CNN_REPORTS_DIR, CNN_TEST_PRED_PATH,
    HANDCRAFTED_PATH, HYBRID_MODEL_PATH, HYBRID_REPORTS_DIR, MODELS_DIR, SEED,
)
from utils import plot_confusion_matrix  # noqa: E402

CONTROL_REPORTS_DIR = os.path.join(HYBRID_REPORTS_DIR, "controle_refit_treino")

META_COLS = ["class", "filename", "split"]
SVM_MODELS = ["SVM_classicas", "SVM_cnn", "SVM_hibrido"]
ALL_MODELS = ["CNN"] + SVM_MODELS

FULL_GRID = {"C": [1, 10, 100], "gamma": ["scale", 1e-3, 1e-2], "pca": [64, 128, 256]}
QUICK_GRID = {"C": [10], "gamma": ["scale"], "pca": [128]}

# Cores das barras do comparativo (slots 1 e 2 da paleta categórica validada)
COLOR_ACCURACY = "#2a78d6"
COLOR_F1 = "#eb6834"
INK_PRIMARY, INK_SECONDARY, GRID_COLOR = "#0b0b0b", "#52514e", "#e4e3df"


# ---------- Dados ----------
def load_features():
    """Junta features clássicas e embeddings (mesma ordem do manifest)."""
    hand = pd.read_csv(HANDCRAFTED_PATH)
    emb = np.load(CNN_EMBEDDINGS_PATH)
    for col in META_COLS:
        if not np.array_equal(hand[col].to_numpy(str), emb[col]):
            raise ValueError(f"handcrafted.csv e cnn_embeddings.npz fora de ordem ({col}). "
                             "Rode extract_features.py de novo.")

    hand_cols = [c for c in hand.columns if c not in META_COLS]
    cnn_cols = [f"cnn_{i:04d}" for i in range(emb["X"].shape[1])]
    data = pd.concat([hand, pd.DataFrame(emb["X"], columns=cnn_cols)], axis=1)
    return data, hand_cols, cnn_cols


# ---------- Modelos ----------
def make_pipeline(model_name, hand_cols, cnn_cols):
    cnn_block = Pipeline([("scaler", StandardScaler()), ("pca", PCA(random_state=SEED))])
    blocks = {
        "SVM_classicas": [("hand", StandardScaler(), hand_cols)],
        "SVM_cnn": [("cnn", cnn_block, cnn_cols)],
        "SVM_hibrido": [("cnn", cnn_block, cnn_cols), ("hand", StandardScaler(), hand_cols)],
    }[model_name]
    return Pipeline([
        ("features", ColumnTransformer(blocks)),
        ("svm", SVC(kernel="rbf", class_weight="balanced")),
    ])


def make_param_grid(model_name, grid, n_fit):
    params = {"svm__C": grid["C"], "svm__gamma": grid["gamma"]}
    if model_name != "SVM_classicas":
        # A PCA não pode ter mais componentes que amostras no ajuste (relevante no --limit)
        valid = [n for n in grid["pca"] if n <= n_fit]
        if not valid:
            valid = [n_fit]
            print(f"  Aviso: só {n_fit} amostras de treino; PCA limitada a {n_fit} componentes.")
        params["features__cnn__pca__n_components"] = valid
    return params


def fit_svm(model_name, data, hand_cols, cnn_cols, grid):
    """GridSearchCV com PredefinedSplit: ajusta no treino, avalia na validação."""
    trval = data[data["split"].isin(["train", "val"])]
    # -1 = sempre no ajuste (treino); 0 = fold de avaliação (validação)
    test_fold = np.where(trval["split"] == "train", -1, 0)
    n_train = int((test_fold == -1).sum())

    search = GridSearchCV(
        make_pipeline(model_name, hand_cols, cnn_cols),
        make_param_grid(model_name, grid, n_train),
        cv=PredefinedSplit(test_fold),
        scoring="f1_macro",
        refit=True,
        n_jobs=-1,
    )
    t0 = time.time()
    search.fit(trval[hand_cols + cnn_cols], trval["class"])
    train_time = time.time() - t0
    print(f"  melhor F1-macro na validação: {search.best_score_:.4f} | {search.best_params_} "
          f"| {train_time:.1f} s")
    return search, train_time


# ---------- Avaliação ----------
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_model_reports(name, y_true, y_pred, class_names, extra, out_dir):
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({"n_test": int(len(y_true))}, **extra)
    with open(os.path.join(out_dir, f"metrics_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(metrics), f, indent=2)

    report = classification_report(y_true, y_pred, labels=class_names,
                                   target_names=class_names, zero_division=0)
    with open(os.path.join(out_dir, f"classification_report_{name}.txt"), "w",
              encoding="utf-8") as f:
        f.write(report)
    print(f"  {name}: accuracy {metrics['accuracy']:.4f} | F1-macro {metrics['f1_macro']:.4f}")
    return metrics


def cnn_train_time():
    """Tempo de treino da CNN, se o metrics_cnn.json do train_model.py existir."""
    path = os.path.join(CNN_REPORTS_DIR, "metrics_cnn.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("train_time_seconds")


def _draw_bars(ax, comparison, ylim, label_offset):
    """Barras agrupadas (accuracy e F1-macro) por modelo, com os valores escritos."""
    models = comparison.index.tolist()
    x = np.arange(len(models))
    width = 0.36
    gap = 0.02  # pequeno espaço entre as duas barras de cada grupo

    for offset, metric, label, color in [
        (-(width + gap) / 2, "accuracy", "Accuracy", COLOR_ACCURACY),
        ((width + gap) / 2, "f1_macro", "F1-macro", COLOR_F1),
    ]:
        values = comparison[metric].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width, label=label, color=color, zorder=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + label_offset, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(models, color=INK_PRIMARY)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)


def plot_comparison(comparison, path, subtitle=""):
    """Dois painéis: escala completa (eixo em 0) e zoom em [0.85, 1.0].

    No painel completo as diferenças entre os modelos (poucos pontos percentuais)
    ficam invisíveis; o zoom mostra essas diferenças, e o título deixa explícito
    que o eixo não começa em 0. Se algum valor ficar abaixo de 0.85, o limite
    inferior do zoom desce o necessário para a barra aparecer.
    """
    zoom_low = 0.85
    lowest = comparison[["accuracy", "f1_macro"]].to_numpy(dtype=float).min()
    if lowest < zoom_low + 0.01:
        zoom_low = np.floor((lowest - 0.02) * 20) / 20

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(15, 5.5))
    _draw_bars(ax_full, comparison, (0, 1.08), 0.01)
    ax_full.set_title("escala completa", color=INK_PRIMARY)
    ax_full.set_ylabel("Valor no conjunto de teste", color=INK_SECONDARY)
    ax_full.legend(frameon=False, loc="upper left", ncol=2)

    _draw_bars(ax_zoom, comparison, (zoom_low, 1.0), 0.002)
    ax_zoom.set_title("zoom (eixo não começa em 0)", color=INK_PRIMARY)

    title = "CNN x SVMs: accuracy e F1-macro no teste"
    if subtitle:
        title += f" ({subtitle})"
    fig.suptitle(title, color=INK_PRIMARY)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Gráfico comparativo salvo em: {path}")


def mcnemar(y_true, pred_a, pred_b, name_a, name_b):
    """Teste de McNemar exato (binomial nos casos discordantes). Retorna o texto."""
    ok_a, ok_b = pred_a == y_true, pred_b == y_true
    n11 = int(np.sum(ok_a & ok_b))     # os dois acertam
    n10 = int(np.sum(ok_a & ~ok_b))    # só A acerta
    n01 = int(np.sum(~ok_a & ok_b))    # só B acerta
    n00 = int(np.sum(~ok_a & ~ok_b))   # os dois erram
    discordant = n10 + n01
    p_value = binomtest(n10, discordant, 0.5).pvalue if discordant > 0 else 1.0

    if p_value < 0.05:
        better = name_a if n10 > n01 else name_b
        verdict = (f"p < 0.05: a diferença é estatisticamente significativa; "
                   f"{better} acerta mais nos casos em que os dois discordam.")
    else:
        verdict = ("p >= 0.05: a diferença NÃO é estatisticamente significativa; "
                   "não dá para dizer que um modelo é melhor que o outro neste teste.")

    w = max(len(name_a), len(name_b)) + 10
    return "\n".join([
        f"=== {name_a} x {name_b} ===",
        f"{'':<{w}}{name_b + ' correto':>22}{name_b + ' errado':>22}",
        f"{name_a + ' correto':<{w}}{n11:>22}{n10:>22}",
        f"{name_a + ' errado':<{w}}{n01:>22}{n00:>22}",
        "",
        f"Casos discordantes: {discordant} (só {name_a} acerta: {n10}; só {name_b} acerta: {n01})",
        f"p-valor (binomial exato, bicaudal): {p_value:.4g}",
        verdict,
        "",
    ])


# ---------- Main ----------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="grade reduzida (C=10, gamma=scale, PCA=128) para teste rápido")
    mode.add_argument("--control-train-only", action="store_true",
                      help="controle: treina com os melhores hiperparâmetros já escolhidos, "
                           "só no split de treino (não refaz a grade)")
    mode.add_argument("--replot", action="store_true",
                      help="só regera reports/hybrid/comparativo.png a partir do comparativo.csv")
    return parser.parse_args()


def load_test_context():
    """Classes, dados, linhas de teste e a predição da CNN no teste."""
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    data, hand_cols, cnn_cols = load_features()
    test = data[data["split"] == "test"]
    print(f"Amostras: {data['split'].value_counts().to_dict()} | "
          f"{len(hand_cols)} features clássicas + {len(cnn_cols)} dims de embedding")

    # CNN (referência): softmax salvo pelo extract_features.py, na ordem das linhas de teste
    cnn_probs = np.load(CNN_TEST_PRED_PATH)
    if len(cnn_probs) != len(test):
        raise ValueError("cnn_test_pred.npy não bate com as linhas de teste. Rode extract_features.py.")
    cnn_pred = np.array(class_names)[cnn_probs.argmax(axis=1)]
    return class_names, data, hand_cols, cnn_cols, test, cnn_pred


def write_reports(out_dir, y_test, predictions, svm_extras, class_names, subtitle=""):
    """Métricas e relatório por modelo, comparativo (csv + png) e McNemar em out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    print("\nAvaliação no TESTE:")
    metrics_all = {"CNN": save_model_reports(
        "CNN", y_test, predictions["CNN"], class_names,
        {"train_time_seconds": cnn_train_time(), "best_params": None}, out_dir)}
    for name in SVM_MODELS:
        metrics_all[name] = save_model_reports(
            name, y_test, predictions[name], class_names, svm_extras[name], out_dir)

    metric_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
                   "train_time_seconds"]
    comparison = pd.DataFrame(
        [[metrics_all[m][k] for k in metric_keys] for m in ALL_MODELS],
        index=pd.Index(ALL_MODELS, name="modelo"), columns=metric_keys, dtype=float)
    comparison.to_csv(os.path.join(out_dir, "comparativo.csv"))
    print("\n" + comparison[["accuracy", "f1_macro", "f1_weighted"]].round(4).to_string())
    plot_comparison(comparison, os.path.join(out_dir, "comparativo.png"), subtitle)

    text = "Teste de McNemar exato no conjunto de TESTE (n = %d)" % len(y_test)
    text += (f" — {subtitle}" if subtitle else "") + "\n\n"
    text += mcnemar(y_test, predictions["CNN"], predictions["SVM_hibrido"], "CNN", "SVM_hibrido")
    text += "\n" + mcnemar(y_test, predictions["SVM_cnn"], predictions["SVM_hibrido"],
                           "SVM_cnn", "SVM_hibrido")
    mcnemar_path = os.path.join(out_dir, "mcnemar.txt")
    with open(mcnemar_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"McNemar salvo em: {mcnemar_path}")


def run_main(quick):
    """Protocolo principal: grade na validação, refit em treino+validação."""
    grid = QUICK_GRID if quick else FULL_GRID
    os.makedirs(MODELS_DIR, exist_ok=True)
    class_names, data, hand_cols, cnn_cols, test, cnn_pred = load_test_context()
    if quick:
        print("Grade reduzida (--quick)")
    y_test = test["class"].to_numpy()
    predictions, svm_extras, searches = {"CNN": cnn_pred}, {}, {}

    os.makedirs(HYBRID_REPORTS_DIR, exist_ok=True)
    for name in SVM_MODELS:
        print(f"\nTreinando {name}...")
        search, train_time = fit_svm(name, data, hand_cols, cnn_cols, grid)
        searches[name] = search
        pd.DataFrame(search.cv_results_).to_csv(
            os.path.join(HYBRID_REPORTS_DIR, f"grid_search_{name}.csv"), index=False)
        predictions[name] = search.predict(test[hand_cols + cnn_cols])
        svm_extras[name] = {"train_time_seconds": round(train_time, 1),
                            "best_params": search.best_params_,
                            "best_val_f1_macro": search.best_score_}

    write_reports(HYBRID_REPORTS_DIR, y_test, predictions, svm_extras, class_names)

    # Matriz de confusão do híbrido (índices na ordem de class_names)
    index = {c: i for i, c in enumerate(class_names)}
    plot_confusion_matrix(
        [index[c] for c in y_test], [index[c] for c in predictions["SVM_hibrido"]], class_names,
        os.path.join(HYBRID_REPORTS_DIR, "confusion_matrix_SVM_hibrido.png"),
        title="Matriz de confusão SVM_hibrido (teste, normalizada por linha)")

    joblib.dump(searches["SVM_hibrido"].best_estimator_, HYBRID_MODEL_PATH)
    print(f"Melhor pipeline híbrido salvo em: {HYBRID_MODEL_PATH}")


def run_control_train_only():
    """Controle: mesmos hiperparâmetros do protocolo principal, ajuste SÓ no treino.

    Não refaz a grade, não sobrescreve reports/hybrid/ nem models/svm_hybrid.joblib.
    """
    class_names, data, hand_cols, cnn_cols, test, cnn_pred = load_test_context()
    train = data[data["split"] == "train"]
    y_test = test["class"].to_numpy()
    predictions, svm_extras = {"CNN": cnn_pred}, {}

    for name in SVM_MODELS:
        metrics_path = os.path.join(HYBRID_REPORTS_DIR, f"metrics_{name}.json")
        if not os.path.exists(metrics_path):
            raise SystemExit(f"{metrics_path} não encontrado: rode o protocolo principal primeiro.")
        with open(metrics_path, encoding="utf-8") as f:
            best_params = json.load(f)["best_params"]

        print(f"\nTreinando {name} só no treino ({len(train)} imagens) com {best_params}...")
        pipe = make_pipeline(name, hand_cols, cnn_cols).set_params(**best_params)
        t0 = time.time()
        pipe.fit(train[hand_cols + cnn_cols], train["class"])
        train_time = time.time() - t0
        print(f"  {train_time:.1f} s")
        predictions[name] = pipe.predict(test[hand_cols + cnn_cols])
        svm_extras[name] = {"train_time_seconds": round(train_time, 1),
                            "best_params": best_params,
                            "protocolo": "refit só no treino (controle)",
                            "n_train": int(len(train))}

    write_reports(CONTROL_REPORTS_DIR, y_test, predictions, svm_extras, class_names,
                  subtitle="controle: SVMs ajustados só no treino")


def replot_main():
    """Regera o gráfico principal a partir do comparativo.csv, sem retreinar."""
    comparison = pd.read_csv(os.path.join(HYBRID_REPORTS_DIR, "comparativo.csv"),
                             index_col="modelo")
    plot_comparison(comparison, os.path.join(HYBRID_REPORTS_DIR, "comparativo.png"))


def main():
    args = parse_args()
    if args.replot:
        replot_main()
    elif args.control_train_only:
        run_control_train_only()
    else:
        run_main(args.quick)


if __name__ == "__main__":
    main()
