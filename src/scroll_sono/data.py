"""Leitura e escrita dos conjuntos de dados."""

import pandas as pd

from scroll_sono.config import CLEAN_DATA_PATH, RAW_DATA_PATH, TEST_DATA_PATH, TRAIN_DATA_PATH


def load_raw() -> pd.DataFrame:
    """Carrega a base bruta exatamente como veio do Kaggle."""
    return pd.read_csv(RAW_DATA_PATH)


def save_clean(df: pd.DataFrame) -> None:
    """Salva a base limpa em data/processed/."""
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)


def load_clean() -> pd.DataFrame:
    """Carrega a base limpa gerada pela etapa de limpeza."""
    return pd.read_csv(CLEAN_DATA_PATH)


def save_train_test(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Salva os conjuntos de treino e de teste em data/processed/."""
    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_DATA_PATH, index=False)
    test.to_csv(TEST_DATA_PATH, index=False)


def load_train() -> pd.DataFrame:
    """Carrega o conjunto de treino gerado pela etapa de separação."""
    return pd.read_csv(TRAIN_DATA_PATH)


def load_test() -> pd.DataFrame:
    """Carrega o conjunto de teste. Usar só na avaliação final."""
    return pd.read_csv(TEST_DATA_PATH)
