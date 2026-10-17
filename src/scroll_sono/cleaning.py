"""Correções determinísticas aplicadas à base bruta.

Nenhuma função aqui calcula estatísticas dos dados (média, moda etc.):
transformações que aprendem com os dados ficam no pré-processamento.
"""

import numpy as np
import pandas as pd

from scroll_sono.config import (BINARY_COLS, CATEGORICAL_COLS, EXPECTED_DOMAINS, ID_COL,
                                NUMERIC_COLS, TARGET)


def standardize_text(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Remove espaços nas pontas e espaços repetidos das colunas de texto.

    Retorna a base corrigida e a quantidade de valores alterados.
    """
    df = df.copy()
    alterados = 0
    for col in cols:
        corrigido = df[col].str.strip().str.replace(r"\s+", " ", regex=True)
        alterados += int((corrigido != df[col]).sum())
        df[col] = corrigido
    return df, alterados


def drop_duplicates(df: pd.DataFrame, id_col: str = ID_COL) -> tuple[pd.DataFrame, int]:
    """Remove linhas repetidas, inclusive as que só diferem pelo identificador.

    Mantém a primeira ocorrência. Retorna a base e a quantidade de linhas removidas.
    """
    repetidas = df.duplicated() | df.drop(columns=id_col).duplicated()
    return df.loc[~repetidas].reset_index(drop=True), int(repetidas.sum())


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Converte as colunas para número; valores que não são números viram NaN.

    Retorna a base e a quantidade de valores que não puderam ser convertidos.
    """
    df = df.copy()
    invalidos = 0
    for col in cols:
        convertido = pd.to_numeric(df[col], errors="coerce")
        invalidos += int((convertido.isna() & df[col].notna()).sum())
        df[col] = convertido
    return df, invalidos


def out_of_domain_to_nan(df: pd.DataFrame, domains: dict = EXPECTED_DOMAINS) -> tuple[pd.DataFrame, int]:
    """Troca por NaN os valores fora do domínio esperado de cada coluna.

    Retorna a base e a quantidade de valores substituídos.
    """
    df = df.copy()
    substituidos = 0
    for col, (minimo, maximo, inteiro) in domains.items():
        s = df[col]
        fora = s < minimo
        if maximo is not None:
            fora |= s > maximo
        if inteiro:
            fora |= s.notna() & (s != s.round())
        substituidos += int(fora.sum())
        df[col] = s.mask(fora, np.nan)
    return df, substituidos


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica todas as correções em sequência.

    Retorna a base limpa e um relatório com a quantidade de alterações de cada passo.
    """
    passos = [
        ("padronizar texto", lambda d: standardize_text(d, CATEGORICAL_COLS + [TARGET, ID_COL])),
        ("remover duplicatas", drop_duplicates),
        ("converter para número", lambda d: coerce_numeric(d, NUMERIC_COLS + BINARY_COLS)),
        ("valores fora do domínio → NaN", out_of_domain_to_nan),
    ]
    linhas = []
    for nome, passo in passos:
        df, alteracoes = passo(df)
        linhas.append({"passo": nome, "alterações": alteracoes, "linhas após o passo": len(df)})
    return df, pd.DataFrame(linhas).set_index("passo")
