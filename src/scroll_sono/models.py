"""Pipelines completos (pré-processamento + modelo) e grades de hiperparâmetros."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from scroll_sono.config import RANDOM_STATE
from scroll_sono.preprocessing import Preprocessor

# Modelos comparados, com o nome usado nas tabelas e gráficos
MODEL_NAMES = {
    "baseline": "Baseline",
    "knn": "KNN",
    "gaussian_nb": "Naive Bayes Gaussiano",
    "categorical_nb": "Naive Bayes Categórico",
    "tree": "Árvore de Decisão",
}

# Grade de hiperparâmetros de cada modelo. As opções do pré-processamento (etapa 5)
# entram como hiperparâmetros do passo "prep".
PARAM_GRIDS = {
    "baseline": {},
    "knn": {
        "prep__log_transform": [False, True],
        "prep__scaler": ["standard", "minmax"],
        "model__n_neighbors": [1, 3, 5, 7, 11, 21, 31, 51, 101],
        "model__weights": ["uniform", "distance"],
    },
    "gaussian_nb": {
        "prep__log_transform": [False, True],
        "model__var_smoothing": np.logspace(-9, 0, 10),
    },
    "categorical_nb": {
        "prep__n_bins": [3, 5, 7, 10, 15, 20],
        "model__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
    },
    "tree": {
        "model__max_depth": [3, 4, 5, 6, 8, 10, 15, None],
        "model__min_samples_leaf": [1, 5, 10, 20, 50, 100],
        "model__class_weight": [None, "balanced"],
    },
}


def _model(name: str):
    if name == "baseline":
        return DummyClassifier(strategy="most_frequent")
    if name == "knn":
        return KNeighborsClassifier()
    if name == "gaussian_nb":
        return GaussianNB()
    if name == "categorical_nb":
        return CategoricalNB()
    if name == "tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    raise ValueError(f"Modelo desconhecido: {name!r}. Use um de {tuple(MODEL_NAMES)}.")


def build_pipeline(name: str, features: list[str], **params) -> Pipeline:
    """Monta o pipeline de um modelo para as colunas de um cenário.

    O baseline usa o pré-processamento da árvore, só para ter a mesma interface; o
    ``DummyClassifier`` ignora os atributos. ``params`` são aplicados com ``set_params``
    (ex.: os melhores hiperparâmetros encontrados pelo ``GridSearchCV``).
    """
    algorithm = "tree" if name == "baseline" else name
    pipeline = Pipeline([("prep", Preprocessor(algorithm, features)), ("model", _model(name))])
    return pipeline.set_params(**params)


def build_pipelines(features: list[str]) -> dict[str, Pipeline]:
    """Um pipeline não ajustado para cada modelo, todos com as colunas do cenário."""
    return {name: build_pipeline(name, features) for name in MODEL_NAMES}
