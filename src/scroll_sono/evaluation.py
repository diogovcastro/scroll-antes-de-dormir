"""Validação cruzada, métricas e gráficos de avaliação."""

import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from scroll_sono.config import CV_SPLITS, RANDOM_STATE, TARGET_CLASSES

# Métricas calculadas na validação cruzada e no teste. A escolha dos hiperparâmetros
# usa o F1 macro (REFIT), que dá o mesmo peso às quatro classes.
SCORING = {"acuracia": "accuracy", "acuracia_balanceada": "balanced_accuracy", "f1_macro": "f1_macro"}
REFIT = "f1_macro"
METRIC_NAMES = {"acuracia": "Acurácia", "acuracia_balanceada": "Acurácia balanceada", "f1_macro": "F1 macro"}

# Cores dos gráficos (mesmo padrão da EDA). O estilo geral (fundo, grade, fontes) fica
# nos notebooks, em plt.rcParams.
SCENARIO_COLORS = {"comportamental": "#2a78d6", "completo": "#eb6834"}
SEQUENTIAL = LinearSegmentedColormap.from_list("sequencial", ["#f0efec", "#86b6ef", "#256abf", "#0d366b"])


def _decimal(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def make_cv() -> StratifiedKFold:
    """Folds estratificados com semente fixa: a mesma divisão para todos os modelos."""
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


@contextmanager
def ignore_bin_warnings():
    """Esconde o aviso esperado da discretização da cafeína (faixas repetidas por causa
    dos 58% de zeros, ver etapa 5)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Bins whose width are too small", category=UserWarning)
        yield


def grid_search(pipeline, param_grid: dict, X: pd.DataFrame, y: pd.Series, cv) -> GridSearchCV:
    """Ajusta os hiperparâmetros por validação cruzada no treino e reajusta o melhor
    pipeline no treino inteiro."""
    search = GridSearchCV(pipeline, param_grid, scoring=SCORING, refit=REFIT, cv=cv, n_jobs=-1)
    with ignore_bin_warnings():
        search.fit(X, y)
    return search


def cv_scores(search: GridSearchCV) -> dict[str, float]:
    """Média e desvio padrão, entre os folds, de cada métrica da melhor combinação."""
    resultados = search.cv_results_
    i = search.best_index_
    scores = {}
    for metrica in SCORING:
        scores[f"cv_{metrica}"] = resultados[f"mean_test_{metrica}"][i]
        scores[f"cv_{metrica}_dp"] = resultados[f"std_test_{metrica}"][i]
    return scores


def test_scores(y_true, y_pred) -> dict[str, float]:
    """Métricas no conjunto de teste."""
    return {
        "teste_acuracia": accuracy_score(y_true, y_pred),
        "teste_acuracia_balanceada": balanced_accuracy_score(y_true, y_pred),
        "teste_f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def plot_cv_comparison(resultados: pd.DataFrame, metrica: str = REFIT, ax=None):
    """Barras horizontais com a média da validação cruzada de cada modelo, por cenário,
    e o desvio padrão entre os folds."""
    ax = ax or plt.gca()
    modelos = list(dict.fromkeys(resultados["modelo"]))
    cenarios = list(dict.fromkeys(resultados["cenário"]))
    altura = 0.8 / len(cenarios)
    cor_texto = plt.rcParams["axes.labelcolor"]
    for j, cenario in enumerate(cenarios):
        dados = resultados[resultados["cenário"] == cenario].set_index("modelo").loc[modelos]
        y = np.arange(len(modelos)) + (j - (len(cenarios) - 1) / 2) * altura
        ax.barh(y, dados[f"cv_{metrica}"], height=altura * 0.85, xerr=dados[f"cv_{metrica}_dp"],
                color=SCENARIO_COLORS.get(cenario), label=cenario,
                error_kw={"lw": 1, "capsize": 2, "ecolor": cor_texto})
        for yi, media, dp in zip(y, dados[f"cv_{metrica}"], dados[f"cv_{metrica}_dp"]):
            ax.text(media + dp + 0.012, yi, _decimal(media), va="center", fontsize=8, color=cor_texto)
    ax.set_yticks(np.arange(len(modelos)), modelos)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(f"{METRIC_NAMES[metrica]} (média dos {CV_SPLITS} folds; a linha é o desvio padrão)")
    ax.grid(axis="y", visible=False)
    ax.legend(title="Cenário", loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=len(cenarios),
              frameon=False)
    return ax


def plot_confusion_matrix(y_true, y_pred, ax=None, title: str | None = None):
    """Matriz de confusão normalizada pelas linhas (cada linha soma 100%), com a
    quantidade de registros entre parênteses."""
    ax = ax or plt.gca()
    contagens = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
    proporcoes = contagens / contagens.sum(axis=1, keepdims=True)
    ax.imshow(proporcoes, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(len(TARGET_CLASSES)):
        for j in range(len(TARGET_CLASSES)):
            cor = "white" if proporcoes[i, j] > 0.5 else plt.rcParams["text.color"]
            ax.text(j, i, f"{_decimal(100 * proporcoes[i, j], 0)}%\n({contagens[i, j]})",
                    ha="center", va="center", fontsize=8, color=cor)
    rotulos = [c.replace(" ", "\n", 1) for c in TARGET_CLASSES]
    ax.set_xticks(range(len(TARGET_CLASSES)), rotulos, fontsize=8)
    ax.set_yticks(range(len(TARGET_CLASSES)), rotulos, fontsize=8)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.grid(False)
    if title:
        ax.set_title(title)
    return ax
