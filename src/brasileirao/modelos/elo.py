"""Rating Elo adaptado ao futebol + logit ordinal para virar probabilidade.

Elo so diz "quem e mais forte". Para virar P(H), P(D), P(A) a diferenca de rating
pre-jogo (ja com o mando somado) entra num LogitOrdinal ajustado no historico.

Ajustes em relacao ao Elo do xadrez (inspirados no World Football Elo):
- vantagem de mando em pontos de rating (`mando`);
- vitoria por 3 gols move mais o rating que vitoria por 1 (multiplicador de saldo);
- na virada de temporada o rating regride parte do caminho para a media, e o
  clube que sobe da Serie B herda a media de quem caiu (ADR-4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brasileirao.metricas import indices
from brasileirao.modelos.base import LogitOrdinal

RATING_INICIAL = 1500.0


class Elo:
    nome = "Elo"

    def __init__(self, k: float = 20.0, mando: float = 70.0, regressao: float = 0.2) -> None:
        self.k = k
        self.mando = mando
        self.regressao = regressao
        self.ratings: dict[str, float] = {}
        self.logit = LogitOrdinal()

    # ------------------------------------------------------------------ rating

    def _virada_de_temporada(self, clubes_novos: set[str], clubes_antigos: set[str]) -> None:
        media = np.mean(list(self.ratings.values())) if self.ratings else RATING_INICIAL
        for c in self.ratings:
            self.ratings[c] += self.regressao * (media - self.ratings[c])
        sairam = clubes_antigos - clubes_novos
        heranca = np.mean([self.ratings[c] for c in sairam]) if sairam else media
        for c in clubes_novos - set(self.ratings):
            self.ratings[c] = heranca
        for c in clubes_novos - clubes_antigos:
            if c in self.ratings:  # voltou depois de anos fora: tambem herda a media de quem caiu
                self.ratings[c] = min(self.ratings[c], heranca)

    def _percorrer(self, historico: pd.DataFrame) -> np.ndarray:
        """Atualiza os ratings jogo a jogo e devolve a diferenca PRE-jogo de cada linha."""
        self.ratings = {}
        diffs = np.empty(len(historico))
        temporada_anterior, clubes_anteriores = None, set()
        h = historico
        temporadas = h["temporada"].to_numpy()
        casas, foras = h["mandante"].to_numpy(), h["visitante"].to_numpy()
        gm, gv = h["gols_mandante"].to_numpy(), h["gols_visitante"].to_numpy()
        clubes_por_temporada = {
            t: set(g["mandante"]) | set(g["visitante"]) for t, g in h.groupby("temporada")
        }

        for i in range(len(h)):
            t = temporadas[i]
            if t != temporada_anterior:
                self._virada_de_temporada(clubes_por_temporada[t], clubes_anteriores)
                temporada_anterior, clubes_anteriores = t, clubes_por_temporada[t]
            rc, rf = self.ratings[casas[i]], self.ratings[foras[i]]
            d = rc + self.mando - rf
            diffs[i] = d
            esperado = 1 / (1 + 10 ** (-d / 400))
            real = 1.0 if gm[i] > gv[i] else 0.5 if gm[i] == gv[i] else 0.0
            saldo = abs(int(gm[i]) - int(gv[i]))
            mult = 1.0 if saldo <= 1 else 1.5 if saldo == 2 else (11 + saldo) / 8
            delta = self.k * mult * (real - esperado)
            self.ratings[casas[i]] = rc + delta
            self.ratings[foras[i]] = rf - delta
        self._clubes_ultima_temporada = clubes_anteriores
        self._ultima_temporada = temporada_anterior
        return diffs

    # ------------------------------------------------------------------ contrato Modelo

    def ajustar(self, historico: pd.DataFrame) -> Elo:
        diffs = self._percorrer(historico)
        # descarta a 1a temporada no ajuste do logit: ratings ainda estao convergindo
        maduro = historico["temporada"].to_numpy() > historico["temporada"].min()
        if maduro.sum() < 200:
            maduro[:] = True
        self.logit.ajustar(diffs[maduro], indices(historico["resultado"])[maduro])
        return self

    def diferenca(self, jogos: pd.DataFrame) -> np.ndarray:
        ratings = dict(self.ratings)
        novos = set(jogos["temporada"]) - {self._ultima_temporada}
        if novos:  # previsao para uma temporada que o historico ainda nao viu
            clubes = set(jogos["mandante"]) | set(jogos["visitante"])
            salvo = self.ratings
            self.ratings = dict(salvo)
            self._virada_de_temporada(clubes, self._clubes_ultima_temporada)
            ratings, self.ratings = self.ratings, salvo
        media = float(np.mean(list(ratings.values())))
        rc = jogos["mandante"].map(ratings).fillna(media).to_numpy(dtype=float)
        rf = jogos["visitante"].map(ratings).fillna(media).to_numpy(dtype=float)
        return rc + self.mando - rf

    def prever(self, jogos: pd.DataFrame) -> np.ndarray:
        return self.logit.prever(self.diferenca(jogos))
