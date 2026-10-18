"""Pré-processamento de cada algoritmo.

``make_preprocessor`` devolve um ``ColumnTransformer`` ainda não ajustado, e
``Preprocessor`` o envolve para que as opções possam ser ajustadas pelo ``GridSearchCV``.
Os dois devem ser colocados dentro de um ``Pipeline`` junto com o modelo, para que o
ajuste (fit) aconteça só nos dados de treino de cada fold.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)
from sklearn.utils.validation import check_is_fitted

from scroll_sono.config import BINARY_COLS, NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS

# Colunas assimétricas à direita (EDA), candidatas à transformação log1p
LOG_COLS = ["bedtime_phone_minutes", "caffeine_post_5pm_mg", "sleep_latency_min"]

ALGORITHMS = ("knn", "gaussian_nb", "categorical_nb", "tree")

# Escalas disponíveis para o KNN
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}

# Codificações das nominais disponíveis para a árvore
NOMINAL_ENCODINGS = ("ordinal", "onehot")


def split_columns(features: list[str]) -> dict[str, list[str]]:
    """Separa as colunas de um cenário pelo tipo, mantendo a ordem do cenário."""
    return {
        "numericas": [c for c in features if c in NUMERIC_COLS],
        "nominais": [c for c in features if c in NOMINAL_COLS],
        "ordinais": [c for c in features if c in ORDINAL_COLS],
        "binarias": [c for c in features if c in BINARY_COLS],
    }


def _log1p() -> FunctionTransformer:
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def _ordinal_encoder(cols: list[str]):
    """Imputação pela moda + codificação ordinal que respeita a ordem natural das categorias."""
    return make_pipeline(SimpleImputer(strategy="most_frequent"),
                         OrdinalEncoder(categories=[ORDINAL_COLS[c] for c in cols]))


def _categorical(encoder):
    """Imputação pela moda seguida do codificador."""
    return make_pipeline(SimpleImputer(strategy="most_frequent"), encoder)


def _numeric_transformers(numericas: list[str], log_transform: bool, final) -> list[tuple]:
    """Transformadores das numéricas: imputação pela mediana, log1p opcional nas colunas
    assimétricas e o passo final do algoritmo."""
    com_log = [c for c in numericas if log_transform and c in LOG_COLS]
    sem_log = [c for c in numericas if c not in com_log]
    passos = []
    if com_log:
        passos.append(("num_log", make_pipeline(SimpleImputer(strategy="median"), _log1p(), final), com_log))
    if sem_log:
        passos.append(("num", make_pipeline(SimpleImputer(strategy="median"), final), sem_log))
    return passos


def make_preprocessor(algorithm: str, features: list[str], log_transform: bool = False,
                      n_bins: int = 5, scaler: str = "standard",
                      nominal_encoding: str = "ordinal") -> ColumnTransformer:
    """Monta o pré-processamento de um algoritmo para as colunas de um cenário.

    - ``knn``: log1p opcional + escala nas numéricas; ordinal + escala no cronotipo;
      one-hot nas nominais. A escala é a padronização (``scaler="standard"``, z-score) ou a
      normalização min-max (``scaler="minmax"``, intervalo [0, 1]).
    - ``gaussian_nb``: log1p opcional nas numéricas; ordinal no cronotipo; one-hot nas nominais.
    - ``categorical_nb``: numéricas discretizadas por frequências iguais (quantis);
      categóricas em códigos inteiros.
    - ``tree``: numéricas sem transformação; cronotipo ordinal; nominais em códigos inteiros
      (``nominal_encoding="ordinal"``) ou one-hot (``nominal_encoding="onehot"``). A árvore
      faz cortes do tipo ``código <= 1,5``, então com códigos inteiros a ordem (alfabética)
      das categorias influencia as divisões; o one-hot permite isolar qualquer categoria
      com um único corte.

    Antes de tudo, os valores ausentes são imputados: mediana nas numéricas (menos sensível
    aos outliers que a média) e moda nas categóricas e na binária. A base atual não tem
    ausentes, mas a imputação protege o pipeline contra valores que a limpeza transforma em
    NaN (ex.: fora do domínio). A coluna binária não tem outra transformação.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo desconhecido: {algorithm!r}. Use um de {ALGORITHMS}.")
    if scaler not in SCALERS:
        raise ValueError(f"Escala desconhecida: {scaler!r}. Use uma de {tuple(SCALERS)}.")
    if nominal_encoding not in NOMINAL_ENCODINGS:
        raise ValueError(f"Codificação desconhecida: {nominal_encoding!r}. Use uma de {NOMINAL_ENCODINGS}.")
    cols = split_columns(features)
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    passos = []

    if algorithm == "knn":
        passos += _numeric_transformers(cols["numericas"], log_transform, SCALERS[scaler]())
        if cols["ordinais"]:
            passos.append(("ord", make_pipeline(_ordinal_encoder(cols["ordinais"]), SCALERS[scaler]()),
                           cols["ordinais"]))
        if cols["nominais"]:
            passos.append(("nom", _categorical(one_hot), cols["nominais"]))

    elif algorithm == "gaussian_nb":
        passos += _numeric_transformers(cols["numericas"], log_transform, "passthrough")
        if cols["ordinais"]:
            passos.append(("ord", _ordinal_encoder(cols["ordinais"]), cols["ordinais"]))
        if cols["nominais"]:
            passos.append(("nom", _categorical(one_hot), cols["nominais"]))

    elif algorithm == "categorical_nb":
        passos += _numeric_transformers(
            cols["numericas"], False, KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile"))
        if cols["ordinais"]:
            passos.append(("ord", _ordinal_encoder(cols["ordinais"]), cols["ordinais"]))
        if cols["nominais"]:
            passos.append(("nom", _categorical(OrdinalEncoder()), cols["nominais"]))

    else:  # tree
        passos += _numeric_transformers(cols["numericas"], False, "passthrough")
        if cols["ordinais"]:
            passos.append(("ord", _ordinal_encoder(cols["ordinais"]), cols["ordinais"]))
        if cols["nominais"]:
            codificador = one_hot if nominal_encoding == "onehot" else OrdinalEncoder()
            passos.append(("nom", _categorical(codificador), cols["nominais"]))

    if cols["binarias"]:
        passos.append(("bin", SimpleImputer(strategy="most_frequent"), cols["binarias"]))

    return ColumnTransformer(passos, remainder="drop", verbose_feature_names_out=False)


class Preprocessor(TransformerMixin, BaseEstimator):
    """Versão de ``make_preprocessor`` com as opções como parâmetros do scikit-learn.

    Assim, ``log_transform``, ``n_bins``, ``scaler`` e ``nominal_encoding`` podem ser
    ajustados pelo ``GridSearchCV`` como qualquer hiperparâmetro (ex.:
    ``prep__log_transform``). O ``ColumnTransformer`` é montado e ajustado a cada ``fit``.
    """

    def __init__(self, algorithm: str = "tree", features: list[str] | None = None,
                 log_transform: bool = False, n_bins: int = 5, scaler: str = "standard",
                 nominal_encoding: str = "ordinal"):
        self.algorithm = algorithm
        self.features = features
        self.log_transform = log_transform
        self.n_bins = n_bins
        self.scaler = scaler
        self.nominal_encoding = nominal_encoding

    def fit(self, X, y=None):
        self.transformer_ = make_preprocessor(self.algorithm, self.features, self.log_transform,
                                              self.n_bins, self.scaler, self.nominal_encoding)
        self.transformer_.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "transformer_")
        return self.transformer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "transformer_")
        return self.transformer_.get_feature_names_out()
