"""Baseline: ignora quem joga e devolve a frequencia historica de H / D / A.

Parece bobo, mas e o piso honesto. Qualquer modelo que nao bate isso nao aprendeu
nada alem de "o mandante vence mais".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brasileirao.metricas import indices


class Frequencia:
    nome = "Frequencia historica"

    def __init__(self) -> None:
        self.p = np.full(3, 1 / 3)

    def ajustar(self, historico: pd.DataFrame) -> Frequencia:
        self.p = np.bincount(indices(historico["resultado"]), minlength=3) / len(historico)
        return self

    def prever(self, jogos: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p, (len(jogos), 1))
