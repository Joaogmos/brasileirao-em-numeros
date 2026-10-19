"""Odds -> probabilidades. O mercado e o benchmark mais duro que existe para o modelo.

A odd decimal embute a margem da casa: 1/odd_H + 1/odd_D + 1/odd_A ~ 1.07 no
Brasileirao (7% de "overround"). Para virar probabilidade e preciso tirar essa
margem. Dois metodos:

- proporcional: divide cada 1/odd pela soma. Simples, mas espalha a margem por
  igual e por isso superestima zebras (vies favorito-azarao).
- Shin (1993): modela a margem como protecao contra apostadores com informacao
  privilegiada (fracao z); tira mais margem das zebras.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def proporcional(odds: np.ndarray) -> np.ndarray:
    q = 1.0 / np.asarray(odds, dtype=float)
    return q / q.sum(axis=1, keepdims=True)


def _shin_linha(q: np.ndarray) -> np.ndarray:
    soma = q.sum()

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4 * (1 - z) * q**2 / soma) - z) / (2 * (1 - z))

    z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.5)
    return probs(z)


def shin(odds: np.ndarray) -> np.ndarray:
    q = 1.0 / np.asarray(odds, dtype=float)
    return np.vstack([_shin_linha(linha) for linha in q])


METODOS = {"proporcional": proporcional, "shin": shin}


def probabilidades_mercado(df, metodo: str = "proporcional") -> np.ndarray:
    odds = df[["odd_mandante", "odd_empate", "odd_visitante"]].to_numpy()
    return METODOS[metodo](odds)
