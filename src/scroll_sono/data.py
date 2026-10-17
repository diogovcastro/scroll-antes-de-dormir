"""Leitura e escrita dos conjuntos de dados."""

import pandas as pd

from scroll_sono.config import RAW_DATA_PATH


def load_raw() -> pd.DataFrame:
    """Carrega a base bruta exatamente como veio do Kaggle."""
    return pd.read_csv(RAW_DATA_PATH)
