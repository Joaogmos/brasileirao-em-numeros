"""Modelo de Dixon & Coles (1997): gols de cada lado como Poisson, com 4 ajustes.

    gols_mandante  ~ Poisson(lam),  lam = exp(mu0 + mando + ataque[casa] + defesa[fora])
    gols_visitante ~ Poisson(mu),   mu  = exp(mu0 +         ataque[fora] + defesa[casa])

1. `defesa` aqui e "fragilidade": quanto maior, mais gols o clube SOFRE.
2. Correcao rho: Poisson independente erra a frequencia de 0x0, 1x0, 0x1 e 1x1
   (placares baixos sao correlacionados). DC multiplica essas 4 celulas por tau.
3. Peso temporal: jogo de ha `t` dias pesa exp(-xi * t). Forca de elenco muda;
   o jogo de ontem diz mais que o de 3 anos atras.
4. Penalidade L2 (ridge) em ataque/defesa -- nao esta no artigo original. Resolve
   a identificabilidade (somar c em todo ataque e tirar de toda defesa da a mesma
   verossimilhanca) e encolhe para a media o clube com poucos jogos (ADR-3).

O ajuste e por maxima verossimilhanca (L-BFGS-B) com gradiente ANALITICO: com
~45 clubes sao ~90 parametros, e gradiente numerico custaria 180 avaliacoes por
iteracao. No walk-forward o modelo e reajustado centenas de vezes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOLS = 10


def tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


class DixonColes:
    nome = "Dixon-Coles"

    def __init__(self, xi: float = 0.002, reg: float = 2.0, janela_dias: int = 1460) -> None:
        self.xi = xi
        self.reg = reg
        self.janela_dias = janela_dias
        self.clubes: list[str] = []
        self.ataque = np.zeros(0)
        self.defesa = np.zeros(0)
        self.mu0 = 0.0
        self.mando = 0.0
        self.rho = 0.0
        self._chute: dict[str, tuple[float, float]] = {}  # warm start entre reajustes

    # ------------------------------------------------------------------ verossimilhanca

    def _nll_e_gradiente(self, theta, ic, ifo, x, y, w, n):
        a, d = theta[:n], theta[n:2 * n]
        mu0, mando, rho = theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]
        loglam = mu0 + mando + a[ic] + d[ifo]
        logmu = mu0 + a[ifo] + d[ic]
        lam, mu = np.exp(loglam), np.exp(logmu)
        t = np.maximum(tau(x, y, lam, mu, rho), 1e-10)

        ll = w * (np.log(t) + x * loglam - lam + y * logmu - mu)
        nll = -ll.sum() + self.reg * (a @ a + d @ d)

        # derivadas de log(tau) em relacao a log(lam), log(mu) e rho
        c00, c01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
        c10, c11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
        dt_lam = np.where(c00, -lam * mu * rho, 0) + np.where(c01, lam * rho, 0)
        dt_mu = np.where(c00, -lam * mu * rho, 0) + np.where(c10, mu * rho, 0)
        dt_rho = (np.where(c00, -lam * mu, 0) + np.where(c01, lam, 0)
                  + np.where(c10, mu, 0) + np.where(c11, -1.0, 0))

        g_lam = w * (x - lam + dt_lam / t)
        g_mu = w * (y - mu + dt_mu / t)
        g = np.empty_like(theta)
        g[:n] = -(np.bincount(ic, g_lam, n) + np.bincount(ifo, g_mu, n)) + 2 * self.reg * a
        g[n:2 * n] = -(np.bincount(ifo, g_lam, n) + np.bincount(ic, g_mu, n)) + 2 * self.reg * d
        g[2 * n] = -(g_lam.sum() + g_mu.sum())
        g[2 * n + 1] = -g_lam.sum()
        g[2 * n + 2] = -(w * dt_rho / t).sum()
        return nll, g

    # ------------------------------------------------------------------ contrato Modelo

    def ajustar(self, historico: pd.DataFrame) -> DixonColes:
        referencia = historico["data"].max()
        idade = (referencia - historico["data"]).dt.days.to_numpy()
        h = historico[idade <= self.janela_dias]
        idade = idade[idade <= self.janela_dias]

        self.clubes = sorted(set(h["mandante"]) | set(h["visitante"]))
        idx = {c: i for i, c in enumerate(self.clubes)}
        n = len(self.clubes)
        ic = h["mandante"].map(idx).to_numpy()
        ifo = h["visitante"].map(idx).to_numpy()
        x = h["gols_mandante"].to_numpy(dtype=float)
        y = h["gols_visitante"].to_numpy(dtype=float)
        w = np.exp(-self.xi * idade)

        theta0 = np.zeros(2 * n + 3)
        for c, i in idx.items():
            theta0[i], theta0[n + i] = self._chute.get(c, (0.0, 0.0))
        theta0[2 * n:] = [self.mu0 or 0.1, self.mando or 0.3, self.rho]

        limites = [(None, None)] * (2 * n + 2) + [(-0.25, 0.25)]
        r = minimize(self._nll_e_gradiente, theta0, args=(ic, ifo, x, y, w, n),
                     jac=True, method="L-BFGS-B", bounds=limites)
        th = r.x
        self.ataque, self.defesa = th[:n], th[n:2 * n]
        self.mu0, self.mando, self.rho = th[2 * n], th[2 * n + 1], th[2 * n + 2]
        self._chute = {c: (self.ataque[i], self.defesa[i]) for c, i in idx.items()}

        # Clube nunca visto (subiu da Serie B pela 1a vez) herda a media dos 4 mais
        # fracos da ultima temporada do historico -- e quem ele substituiu (ADR-4).
        ultima = h[h["temporada"] == h["temporada"].max()]
        recentes = [idx[c] for c in set(ultima["mandante"]) | set(ultima["visitante"])]
        forca = self.ataque[recentes] - self.defesa[recentes]
        fracos = np.array(recentes)[np.argsort(forca)[:4]]
        self._novato = (float(self.ataque[fracos].mean()), float(self.defesa[fracos].mean()))
        return self

    def taxas(self, jogos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Gols esperados (lam, mu) de cada jogo."""
        idx = {c: i for i, c in enumerate(self.clubes)}

        def params(clubes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
            i = clubes.map(idx)
            conhecido = i.notna().to_numpy()
            ii = i.fillna(0).astype(int).to_numpy()
            a = np.where(conhecido, self.ataque[ii], self._novato[0])
            d = np.where(conhecido, self.defesa[ii], self._novato[1])
            return a, d

        ac, dc = params(jogos["mandante"])
        af, df = params(jogos["visitante"])
        lam = np.exp(self.mu0 + self.mando + ac + df)
        mu = np.exp(self.mu0 + af + dc)
        return lam, mu

    def matriz_placares(self, lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """P(placar = i x j) para cada jogo -> array (n_jogos, 11, 11)."""
        g = np.arange(MAX_GOLS + 1)
        pc = poisson.pmf(g[None, :], lam[:, None])
        pf = poisson.pmf(g[None, :], mu[:, None])
        m = pc[:, :, None] * pf[:, None, :]
        r = self.rho
        m[:, 0, 0] *= 1 - lam * mu * r
        m[:, 0, 1] *= 1 + lam * r
        m[:, 1, 0] *= 1 + mu * r
        m[:, 1, 1] *= 1 - r
        return m / m.sum(axis=(1, 2), keepdims=True)

    def prever(self, jogos: pd.DataFrame) -> np.ndarray:
        m = self.matriz_placares(*self.taxas(jogos))
        p_h = np.tril(m, -1).sum(axis=(1, 2))
        p_d = np.trace(m, axis1=1, axis2=2)
        p_a = np.triu(m, 1).sum(axis=(1, 2))
        return np.column_stack([p_h, p_d, p_a])

    def forcas(self) -> pd.DataFrame:
        """Tabela legivel dos parametros: gols marcados/sofridos esperados contra um time medio."""
        return pd.DataFrame({
            "clube": self.clubes,
            "ataque": self.ataque,
            "defesa": self.defesa,
            "gols_pro_esperados": np.exp(self.mu0 + self.ataque + self.mando / 2),
            "gols_contra_esperados": np.exp(self.mu0 + self.defesa + self.mando / 2),
        }).assign(forca=lambda f: f["ataque"] - f["defesa"]).sort_values("forca", ascending=False)
