"""Classificacao da Serie A com os criterios de desempate da CBF.

Criterios (Regulamento Especifico da Serie A): 1) pontos, 2) vitorias,
3) saldo de gols, 4) gols marcados. Os seguintes (confronto direto, cartoes,
sorteio) quase nunca decidem; aqui o empate restante e resolvido por ordem
alfabetica (na tabela real) ou por sorteio (na simulacao).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CRITERIOS = ["pontos", "vitorias", "saldo", "gols_pro"]


def classificacao(partidas: pd.DataFrame) -> pd.DataFrame:
    casa = pd.DataFrame({
        "clube": partidas["mandante"],
        "gols_pro": partidas["gols_mandante"],
        "gols_contra": partidas["gols_visitante"],
    })
    fora = pd.DataFrame({
        "clube": partidas["visitante"],
        "gols_pro": partidas["gols_visitante"],
        "gols_contra": partidas["gols_mandante"],
    })
    linhas = pd.concat([casa, fora], ignore_index=True)
    linhas["vitorias"] = (linhas["gols_pro"] > linhas["gols_contra"]).astype(int)
    linhas["empates"] = (linhas["gols_pro"] == linhas["gols_contra"]).astype(int)
    linhas["derrotas"] = (linhas["gols_pro"] < linhas["gols_contra"]).astype(int)
    linhas["jogos"] = 1

    t = linhas.groupby("clube").sum()
    t["pontos"] = 3 * t["vitorias"] + t["empates"]
    t["saldo"] = t["gols_pro"] - t["gols_contra"]
    t = t.reset_index().sort_values(CRITERIOS + ["clube"], ascending=[False] * 4 + [True])
    t.insert(0, "posicao", np.arange(1, len(t) + 1))
    colunas = ["posicao", "clube", "pontos", "jogos", "vitorias", "empates", "derrotas",
               "gols_pro", "gols_contra", "saldo"]
    return t[colunas].reset_index(drop=True)
