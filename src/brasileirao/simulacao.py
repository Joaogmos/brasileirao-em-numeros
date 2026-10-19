"""Simulacao de Monte Carlo do restante de uma temporada.

Para cada jogo que falta, o Dixon-Coles da a matriz P(placar = i x j). Sorteamos
um placar por jogo, somamos a tabela atual, ordenamos pelos criterios da CBF e
repetimos N vezes. A frequencia com que cada clube termina em cada posicao e a
probabilidade estimada (ADR-6).

Simular PLACARES (e nao so H/D/A) importa: saldo e gols pro desempatam a tabela.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from brasileirao import config
from brasileirao.modelos.dixon_coles import MAX_GOLS, DixonColes
from brasileirao.tabela import classificacao


def jogos_restantes(jogados: pd.DataFrame, clubes: list[str] | None = None) -> pd.DataFrame:
    """Turno e returno completo = todo par ordenado (mandante, visitante) uma vez.

    A fonte so traz jogos ja disputados, mas o calendario restante e deduzivel:
    sao os pares ordenados que ainda nao aparecem.
    """
    clubes = clubes or sorted(set(jogados["mandante"]) | set(jogados["visitante"]))
    feitos = set(zip(jogados["mandante"], jogados["visitante"], strict=True))
    faltam = [(c, f) for c, f in itertools.permutations(clubes, 2) if (c, f) not in feitos]
    return pd.DataFrame(faltam, columns=["mandante", "visitante"])


def simular(modelo: DixonColes, jogados: pd.DataFrame, restantes: pd.DataFrame,
            n: int = 10000, semente: int = config.SEMENTE) -> dict:
    rng = np.random.default_rng(semente)
    atual = classificacao(jogados) if len(jogados) else None
    clubes = sorted(set(restantes["mandante"]) | set(restantes["visitante"])
                    | (set(atual["clube"]) if atual is not None else set()))
    idx = {c: i for i, c in enumerate(clubes)}
    k = len(clubes)

    base = np.zeros((4, k))  # pontos, vitorias, saldo, gols_pro
    if atual is not None:
        a = atual.set_index("clube").reindex(clubes).fillna(0)
        base = a[["pontos", "vitorias", "saldo", "gols_pro"]].to_numpy().T.astype(float)

    pontos = np.tile(base[0], (n, 1))
    vitorias = np.tile(base[1], (n, 1))
    saldo = np.tile(base[2], (n, 1))
    gols_pro = np.tile(base[3], (n, 1))

    if len(restantes):
        m = modelo.matriz_placares(*modelo.taxas(restantes)).reshape(len(restantes), -1)
        cdf = np.cumsum(m, axis=1)
        u = rng.random((n, len(restantes)))
        lado = MAX_GOLS + 1
        for j, (c, f) in enumerate(zip(restantes["mandante"], restantes["visitante"], strict=True)):
            cel = np.minimum(np.searchsorted(cdf[j], u[:, j]), lado * lado - 1)
            gc, gf = cel // lado, cel % lado
            ic, ifo = idx[c], idx[f]
            pontos[:, ic] += np.select([gc > gf, gc == gf], [3, 1], 0)
            pontos[:, ifo] += np.select([gf > gc, gc == gf], [3, 1], 0)
            vitorias[:, ic] += gc > gf
            vitorias[:, ifo] += gf > gc
            saldo[:, ic] += gc - gf
            saldo[:, ifo] += gf - gc
            gols_pro[:, ic] += gc
            gols_pro[:, ifo] += gf

    # ordenacao lexicografica pelos criterios da CBF; sorteio desfaz o resto
    ordem = np.lexsort((rng.random((n, k)), -gols_pro, -saldo, -vitorias, -pontos), axis=1)
    posicao = np.empty_like(ordem)
    np.put_along_axis(posicao, ordem, np.arange(1, k + 1)[None, :].repeat(n, 0), axis=1)

    dist = pd.DataFrame(
        np.stack([(posicao == p).mean(axis=0) for p in range(1, k + 1)], axis=1),
        index=clubes, columns=range(1, k + 1),
    )
    resumo = pd.DataFrame({
        "pontos_atuais": base[0],
        "pontos_esperados": pontos.mean(axis=0),
        "pontos_p05": np.quantile(pontos, 0.05, axis=0),
        "pontos_p95": np.quantile(pontos, 0.95, axis=0),
        "posicao_media": posicao.mean(axis=0),
    }, index=clubes)
    return {"posicoes": dist, "resumo": resumo, "_pontos": pontos, "_posicao": posicao, "clubes": clubes}


def zonas(dist: pd.DataFrame, g: int = 6, z: int = 4) -> pd.DataFrame:
    k = dist.shape[1]
    return pd.DataFrame({
        "titulo": dist[1],
        f"top{g}": dist[list(range(1, g + 1))].sum(axis=1),
        "rebaixamento": dist[list(range(k - z + 1, k + 1))].sum(axis=1),
    })
