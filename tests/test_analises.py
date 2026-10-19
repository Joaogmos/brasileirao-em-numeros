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
