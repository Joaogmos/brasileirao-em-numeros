import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from brasileirao.dados import limpar
from brasileirao.modelos.dixon_coles import DixonColes, tau
from brasileirao.modelos.elo import Elo
from brasileirao.modelos.frequencia import Frequencia


def liga_com_forcas_conhecidas(n_clubes=16, temporadas=3, semente=0):
    """Simula uma liga a partir de parametros VERDADEIROS para ver se o DC os recupera."""
    rng = np.random.default_rng(semente)
    clubes = [f"C{i}" for i in range(n_clubes)]
    ataque = rng.normal(0, 0.3, n_clubes)
    defesa = rng.normal(0, 0.3, n_clubes)
    mu0, mando = 0.1, 0.35
    linhas, dia = [], pd.Timestamp("2030-01-01")
    for t in range(temporadas):
        for (i, a), (j, b) in itertools.permutations(enumerate(clubes), 2):
            lam = np.exp(mu0 + mando + ataque[i] + defesa[j])
            mu = np.exp(mu0 + ataque[j] + defesa[i])
            x, y = rng.poisson(lam), rng.poisson(mu)
            linhas.append({"temporada": 2030 + t, "data": dia, "mandante": a, "visitante": b,
                           "gols_mandante": x, "gols_visitante": y,
                           "resultado": "H" if x > y else "A" if x < y else "D"})
            dia += pd.Timedelta(hours=8)
    return pd.DataFrame(linhas), ataque, defesa, mando


def test_gradiente_analitico_bate_com_o_numerico():
    df, *_ = liga_com_forcas_conhecidas(n_clubes=6, temporadas=1)
    m = DixonColes(reg=0.5)
    idx = {c: i for i, c in enumerate(sorted(set(df["mandante"])))}
    args = (df["mandante"].map(idx).to_numpy(), df["visitante"].map(idx).to_numpy(),
            df["gols_mandante"].to_numpy(float), df["gols_visitante"].to_numpy(float),
            np.linspace(0.3, 1, len(df)), len(idx))
    theta = np.random.default_rng(3).normal(0, 0.2, 2 * len(idx) + 3)
    theta[-1] = 0.05
    _, analitico = m._nll_e_gradiente(theta, *args)
    numerico = approx_fprime(theta, lambda t: m._nll_e_gradiente(t, *args)[0], 1e-6)
    assert np.allclose(analitico, numerico, rtol=1e-4, atol=1e-3)


def test_dixon_coles_recupera_parametros_verdadeiros():
    df, ataque, defesa, mando = liga_com_forcas_conhecidas(temporadas=8)
    m = DixonColes(xi=0.0, reg=0.01).ajustar(df)
    ordem = [int(c[1:]) for c in m.clubes]
    assert np.corrcoef(m.ataque, ataque[ordem])[0, 1] > 0.9
    assert np.corrcoef(m.defesa, defesa[ordem])[0, 1] > 0.9
    assert m.mando == pytest.approx(mando, abs=0.08)


def test_previsoes_sao_distribuicoes_validas():
    df, *_ = liga_com_forcas_conhecidas(temporadas=2)
    for modelo in (DixonColes(), Elo(), Frequencia()):
        p = modelo.ajustar(df).prever(df.tail(50))
        assert p.shape == (50, 3)
        assert np.allclose(p.sum(axis=1), 1) and (p >= 0).all()


def test_tau_so_mexe_nos_placares_baixos():
    x = np.array([0, 0, 1, 1, 2, 3])
    y = np.array([0, 1, 0, 1, 2, 0])
    t = tau(x, y, np.full(6, 1.5), np.full(6, 1.0), 0.1)
    assert (t[:4] != 1).all() and (t[4:] == 1).all()


def test_clube_desconhecido_recebe_parametros_de_clube_fraco():
    df, *_ = liga_com_forcas_conhecidas(temporadas=2)
    m = DixonColes().ajustar(df)
    jogo = pd.DataFrame({"mandante": ["Novato"], "visitante": [m.forcas()["clube"].iloc[0]]})
    p = m.prever(jogo)[0]
    assert p[2] > p[0]  # o melhor clube da liga e favorito mesmo fora de casa contra o novato


def test_elo_e_soma_zero_e_premia_quem_vence(bruto):
    partidas = limpar(bruto).partidas
    elo = Elo().ajustar(partidas)
    assert np.mean(list(elo.ratings.values())) == pytest.approx(1500, abs=1e-6)
    um_jogo = partidas.iloc[[0]].assign(gols_mandante=3, gols_visitante=0, resultado="H")
    elo._percorrer(um_jogo)
    assert elo.ratings[um_jogo["mandante"].iloc[0]] > 1500


def test_elo_mando_favorece_o_mandante_entre_iguais(bruto):
    partidas = limpar(bruto).partidas
    elo = Elo().ajustar(partidas)
    elo.ratings = dict.fromkeys(elo.ratings, 1500.0)
    p = elo.prever(partidas.tail(1))[0]
    assert p[0] > p[2]


def test_frequencia_devolve_frequencias():
    df = pd.DataFrame({"resultado": ["H", "H", "D", "A"]})
    assert Frequencia().ajustar(df).prever(df)[0].tolist() == [0.5, 0.25, 0.25]
