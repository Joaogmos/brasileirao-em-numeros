import pandas as pd
import pytest

from brasileirao.analises.mando import portoes_fechados
from brasileirao.dados import limpar
from tests.conftest import temporada_sintetica


@pytest.fixture(scope="module")
def nove_temporadas():
    clubes = [f"C{i:02d}" for i in range(20)]
    bruto = pd.concat([temporada_sintetica(clubes, t, semente=t) for t in range(2012, 2021)])
    return limpar(bruto).partidas


def test_portoes_fechados_compara_2020_com_2012_2019(nove_temporadas):
    p = nove_temporadas
    r = portoes_fechados(p, n_boot=500)
    saldo = p["gols_mandante"] - p["gols_visitante"]
    assert r["saldo_sem_publico"] == pytest.approx(saldo[p["temporada"] == 2020].mean())
    assert r["saldo_com_publico"] == pytest.approx(saldo[p["temporada"] < 2020].mean())
    assert r["ic95"][0] <= r["diferenca"] <= r["ic95"][1]
    # dados sinteticos tem o MESMO mando em todas as temporadas: nao pode dar "significativo"
    assert r["p_valor_permutacao"] > 0.01


def test_xpts_bate_pontos_da_tabela_e_soma_esperada(nove_temporadas):
    import numpy as np

    from brasileirao import odds
    from brasileirao.analises.santos import resumo_xpts
    from brasileirao.tabela import classificacao

    t = nove_temporadas[nove_temporadas["temporada"] == 2015].reset_index(drop=True)
    probs = odds.probabilidades_mercado(t)
    r = resumo_xpts(t, probs).set_index("clube")
    tabela = classificacao(t).set_index("clube")
    assert (r["pontos"] == tabela["pontos"].reindex(r.index)).all()
    # cada jogo distribui 3*P(H)+P(D) + 3*P(A)+P(D) = 3 - P(D) pontos esperados
    assert r["xpts"].sum() == pytest.approx((3 - probs[:, 1]).sum())
    assert np.isfinite(r["diferenca"]).all()


def test_simular_pontos_do_mercado_tem_a_media_do_xpts(nove_temporadas):
    from brasileirao import odds
    from brasileirao.analises.santos import resumo_xpts, simular_pontos

    t = nove_temporadas[nove_temporadas["temporada"] == 2015].reset_index(drop=True)
    probs = odds.probabilidades_mercado(t)
    clubes, pontos, posicao = simular_pontos(t, probs, n=4000)
    esperado = resumo_xpts(t, probs).set_index("clube").loc[clubes, "xpts"].to_numpy()
    assert abs(pontos.mean(axis=0) - esperado).max() < 0.5
    assert sorted(posicao[0]) == list(range(1, 21))
