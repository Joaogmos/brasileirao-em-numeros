import pandas as pd
import pytest

from brasileirao import config
from brasileirao.dados import ErroDeValidacao, carregar, limpar, validar


def test_temporada_sintetica_completa_passa(bruto):
    dados = limpar(bruto)
    validar(dados)
    assert len(dados.partidas) == 760
    assert dados.pendentes.empty


def test_resultado_inconsistente_com_placar_e_rejeitado(bruto):
    bruto.loc[0, ["HG", "AG", "Res"]] = [2, 0, "A"]
    with pytest.raises(ErroDeValidacao, match="diferente do placar"):
        validar(limpar(bruto))


def test_confronto_repetido_e_rejeitado(bruto):
    duplicado = pd.concat([bruto, bruto.iloc[[0]]], ignore_index=True)
    with pytest.raises(ErroDeValidacao, match="repetido"):
        validar(limpar(duplicado))


def test_odd_absurda_e_rejeitada(bruto):
    bruto.loc[5, "AvgCH"] = 0.9
    with pytest.raises(ErroDeValidacao, match="odd"):
        validar(limpar(bruto))


def test_temporada_encerrada_incompleta_e_rejeitada(bruto):
    sem_um_jogo = bruto[bruto["Season"].eq(2031) | (bruto.index != 0)]
    with pytest.raises(ErroDeValidacao, match="2030: 379 jogos"):
        validar(limpar(sem_um_jogo))


def test_jogo_sem_placar_vira_pendente_e_fecha_a_conta(bruto):
    bruto.loc[0, ["HG", "AG", "Res"]] = [None, None, None]
    dados = limpar(bruto)
    validar(dados)  # 379 + 1 pendente = 380
    assert len(dados.pendentes) == 1


def test_temporada_atual_pode_estar_incompleta(bruto):
    em_andamento = bruto[(bruto["Season"] == 2030) | (bruto.index < 400)]
    validar(limpar(em_andamento))


def test_validar_junta_todos_os_problemas(bruto):
    bruto.loc[0, "Res"] = "X"
    bruto.loc[1, "AvgCD"] = 1.0
    with pytest.raises(ErroDeValidacao) as exc:
        validar(limpar(bruto))
    assert len(exc.value.problemas) == 2


def test_nomes_normalizados(bruto):
    bruto.loc[0, "Home"] = "Flamengo RJ"
    assert "Flamengo" in set(limpar(bruto).partidas["mandante"])


@pytest.mark.skipif(not config.ARQUIVO_BRUTO.exists(), reason="snapshot ausente")
def test_snapshot_real_bate_com_manifesto_e_e_valido():
    dados = carregar()
    p = dados.partidas
    assert p["temporada"].min() == 2012
    # o unico jogo nao realizado da historia recente: Chapecoense x Atletico-MG, 2016
    assert dados.pendentes[["mandante", "visitante"]].values.tolist() == [["Chapecoense", "Atletico-MG"]]
    assert p["data"].is_monotonic_increasing
