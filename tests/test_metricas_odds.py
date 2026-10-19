import numpy as np
import pytest

from brasileirao import metricas, odds


def test_rps_valores_conhecidos():
    y = np.array([0])  # mandante venceu
    assert metricas.rps(np.array([[1.0, 0.0, 0.0]]), y)[0] == 0.0
    assert metricas.rps(np.array([[0.0, 0.0, 1.0]]), y)[0] == 1.0
    # exemplo de Constantinou & Fenton (2012): (1, 0, 0) vs previsao (.6, .3, .1)
    assert metricas.rps(np.array([[0.6, 0.3, 0.1]]), y)[0] == pytest.approx(0.085)


def test_rps_respeita_a_ordem_e_logloss_nao():
    y = np.array([0])
    errou_por_um = np.array([[0.2, 0.6, 0.2]])   # apostou no empate
    errou_por_dois = np.array([[0.2, 0.2, 0.6]])  # apostou no visitante
    assert metricas.rps(errou_por_um, y) < metricas.rps(errou_por_dois, y)
    assert metricas.log_loss(errou_por_um, y) == pytest.approx(metricas.log_loss(errou_por_dois, y))


def test_log_loss_nao_explode_com_probabilidade_zero():
    assert np.isfinite(metricas.log_loss(np.array([[0.0, 0.0, 1.0]]), np.array([0]))).all()


def test_bootstrap_detecta_diferenca_real_e_nao_inventa_diferenca():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.1, 0.3, 3000)
    melhor = metricas.bootstrap_diferenca(base - 0.01, base)
    assert melhor["ic_sup"] < 0 and melhor["p_a_melhor"] == 1.0
    igual = metricas.bootstrap_diferenca(base + rng.normal(0, 0.01, 3000), base)
    assert igual["ic_inf"] < 0 < igual["ic_sup"]


def test_calibracao_de_previsao_perfeita_fica_na_diagonal():
    rng = np.random.default_rng(2)
    p = rng.dirichlet([2, 2, 2], size=20000)
    y = np.array([rng.choice(3, p=linha) for linha in p])
    cal = metricas.calibracao(p, y)
    assert np.allclose(cal["prevista"], cal["observada"], atol=0.03)


@pytest.mark.parametrize("metodo", ["proporcional", "shin"])
def test_odds_viram_probabilidades_que_somam_1(metodo):
    o = np.array([[1.50, 4.20, 7.00], [2.90, 3.10, 2.60]])
    p = odds.METODOS[metodo](o)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p.argmax(axis=1) == (1 / o).argmax(axis=1)).all()


def test_shin_tira_mais_margem_da_zebra_que_o_proporcional():
    o = np.array([[1.30, 5.50, 11.0]])
    assert odds.shin(o)[0, 2] < odds.proporcional(o)[0, 2]
    assert odds.shin(o)[0, 0] > odds.proporcional(o)[0, 0]
