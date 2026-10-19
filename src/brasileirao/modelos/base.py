"""Contrato comum dos modelos + regressao logistica ordinal usada pelo Elo."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


class Modelo(Protocol):
    """Todo modelo aprende SO com o passado e devolve P(H), P(D), P(A) por jogo.

    `ajustar` recebe apenas jogos anteriores ao momento da previsao: quem garante
    isso e o backtest (walk-forward), nunca o modelo.
    """

    nome: str

    def ajustar(self, historico: pd.DataFrame) -> Modelo: ...

    def prever(self, jogos: pd.DataFrame) -> np.ndarray: ...


class LogitOrdinal:
    """P(resultado | x) com resultados ordenados A < D < H e um unico coeficiente.

        P(A)      = sigma(c1 - b*x)
        P(A ou D) = sigma(c2 - b*x),  c1 < c2

    Tres parametros. Uma logistica multinomial comum teria 4 e ignoraria que o
    empate fica "entre" as vitorias -- com uma variavel so (diferenca de forca), a
    estrutura ordinal e a hipotese natural.
    """

    def __init__(self) -> None:
        self.b, self.c1, self.c2 = 0.0, -1.0, 0.5

    @staticmethod
    def _probs(x: np.ndarray, b: float, c1: float, c2: float) -> np.ndarray:
        pa = expit(c1 - b * x)
        pad = expit(c2 - b * x)
        return np.column_stack([1 - pad, pad - pa, pa])  # ordem H, D, A

    def ajustar(self, x: np.ndarray, y: np.ndarray) -> LogitOrdinal:
        def nll(theta: np.ndarray) -> float:
            b, c1, log_gap = theta
            p = self._probs(x, b, c1, c1 + np.exp(log_gap))
            return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).sum()

        r = minimize(nll, x0=[0.0, -1.0, 0.0], method="L-BFGS-B")
        self.b, self.c1, self.c2 = r.x[0], r.x[1], r.x[1] + np.exp(r.x[2])
        return self

    def prever(self, x: np.ndarray) -> np.ndarray:
        return self._probs(np.asarray(x, dtype=float), self.b, self.c1, self.c2)
