import numpy as np
import pandas as pd

from brasileirao.modelos.dixon_coles import DixonColes
from brasileirao.simulacao import jogos_restantes, simular, zonas
from brasileirao.tabela import classificacao


def _jogos(linhas):
    return pd.DataFrame(linhas, columns=["mandante", "visitante", "gols_mandante", "gols_visitante"])


def test_classificacao_aplica_desempates_da_cbf():
    # A e B com 3 pontos e 1 vitoria; A tem saldo melhor. C e D: mesmos pontos e vitorias,
    # mas D tem saldo melhor (-1 contra -3).
    jogos = _jogos([("A", "C", 3, 0), ("B", "D", 1, 0), ("C", "D", 2, 2), ("D", "C", 1, 1)])
    t = classificacao(jogos)
    assert t["clube"].tolist() == ["A", "B", "D", "C"]
    assert t.set_index("clube").loc["A", ["pontos", "saldo", "jogos"]].tolist() == [3, 3, 1]


def test_vitorias_desempatam_antes_do_saldo():
    # X: 1V 0E 1D (3 pts, saldo -4); Y: 0V 3E 0D (3 pts, saldo 0) -> X na frente
    jogos = _jogos([("X", "Z", 1, 0), ("W", "X", 5, 0), ("Y", "Z", 0, 0), ("Y", "W", 1, 1), ("Z", "Y", 2, 2)])
    t = classificacao(jogos).set_index("clube")
    assert t.loc["X", "posicao"] < t.loc["Y", "posicao"]


def test_jogos_restantes_completa_turno_e_returno():
    jogados = _jogos([("A", "B", 1, 0), ("B", "C", 0, 0)])
    r = jogos_restantes(jogados)
    assert len(r) == 3 * 2 - 2
    assert ("A", "B") not in set(zip(r["mandante"], r["visitante"], strict=True))


def test_simulacao_respeita_tabela_atual_e_probabilidades_somam_1():
    rng = np.random.default_rng(0)
    clubes = list("ABCDEF")
    linhas = [(c, f, rng.poisson(1.4), rng.poisson(1.0)) for c in clubes for f in clubes if c != f] * 3
    hist = _jogos(linhas).assign(data=pd.date_range("2030-01-01", periods=len(linhas), freq="D"),
                                 temporada=2030)
    hist["resultado"] = "D"
    modelo = DixonColes().ajustar(hist)

    jogados = _jogos([(c, f, 5, 0) if c == "A" else (c, f, 0, 0) for c in clubes for f in clubes
                      if c != f and clubes.index(c) < clubes.index(f)])
    sim = simular(modelo, jogados, jogos_restantes(jogados, clubes), n=2000)
    dist = sim["posicoes"]
    assert np.allclose(dist.sum(axis=0), 1) and np.allclose(dist.sum(axis=1), 1)
    assert (sim["_pontos"] >= sim["resumo"]["pontos_atuais"].to_numpy()).all()
    assert zonas(dist, g=2, z=1).loc["A", "titulo"] > 0.5  # A largou com 5 vitorias
