"""Metricas para previsoes probabilisticas de 3 resultados ordenados (H > D > A).

Acuracia nao serve como metrica principal: um modelo que diz 34% / 33% / 33% e
outro que diz 90% / 5% / 5% "acertam" o mesmo jogo, mas so um deles e util. Aqui
o que importa e a QUALIDADE da probabilidade -> regras de pontuacao proprias
(proper scoring rules): RPS, log loss e Brier. Todas: menor = melhor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brasileirao.config import RESULTADOS

_EPS = 1e-15


def indices(resultados: pd.Series | np.ndarray) -> np.ndarray:
    """'H'/'D'/'A' -> 0/1/2 (a ordem importa para o RPS)."""
    mapa = {r: i for i, r in enumerate(RESULTADOS)}
    return np.array([mapa[r] for r in np.asarray(resultados)])


def _one_hot(y: np.ndarray, k: int = 3) -> np.ndarray:
    return np.eye(k)[y]


def rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked Probability Score por jogo (Epstein, 1969; Constantinou & Fenton, 2012).

    Compara as distribuicoes ACUMULADAS. Por isso respeita a ordem dos resultados:
    dar 60% ao empate quando o mandante venceu e "menos errado" do que dar 60% a
    vitoria do visitante. Log loss e Brier tratam os dois erros como iguais.
    """
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_o = np.cumsum(_one_hot(y, p.shape[1]), axis=1)[:, :-1]
    return ((cum_p - cum_o) ** 2).sum(axis=1) / (p.shape[1] - 1)


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], _EPS, 1.0))


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((p - _one_hot(y, p.shape[1])) ** 2).sum(axis=1)


def acerto(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p.argmax(axis=1) == y).astype(float)


METRICAS = {"rps": rps, "log_loss": log_loss, "brier": brier, "acuracia": acerto}


def resumo(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {nome: float(f(p, y).mean()) for nome, f in METRICAS.items()}


def bootstrap_diferenca(
    a: np.ndarray, b: np.ndarray, n: int = 5000, semente: int = 0, alfa: float = 0.05
) -> dict[str, float]:
    """Bootstrap PAREADO da diferenca media a - b (mesmos jogos nas duas amostras).

    Pareado porque os dois modelos foram avaliados nos mesmos jogos: reamostrar os
    jogos e calcular a diferenca dentro de cada reamostra elimina a variancia que
    vem de "jogos dificeis" e deixa so a variancia da diferenca entre os modelos.
    """
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(semente)
    medias = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.quantile(medias, [alfa / 2, 1 - alfa / 2])
    return {"diferenca": float(d.mean()), "ic_inf": float(lo), "ic_sup": float(hi),
            "p_a_melhor": float((medias < 0).mean())}


def calibracao(p: np.ndarray, y: np.ndarray, n_faixas: int = 10) -> pd.DataFrame:
    """Diagrama de confiabilidade: quando o modelo diz X%, o evento acontece X% das vezes?

    Empilha os 3 resultados (cada jogo gera 3 pares previsao/aconteceu) e agrupa
    as previsoes em faixas de mesma largura.
    """
    prev = p.ravel()
    ocorreu = _one_hot(y, p.shape[1]).ravel()
    faixa = np.minimum((prev * n_faixas).astype(int), n_faixas - 1)
    df = pd.DataFrame({"faixa": faixa, "prevista": prev, "observada": ocorreu})
    return (df.groupby("faixa")
              .agg(prevista=("prevista", "mean"), observada=("observada", "mean"), n=("observada", "size"))
              .reset_index(drop=True))
