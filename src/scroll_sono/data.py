"""Leitura e escrita dos conjuntos de dados."""

import pandas as pd

from scroll_sono.config import CLEAN_DATA_PATH, RAW_DATA_PATH


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
