import numpy as np
import pandas as pd

from brasileirao.backtest import walk_forward
from brasileirao.dados import limpar


class Espiao:
    """Modelo falso que registra o que viu: prova que o walk-forward nao vaza o futuro."""

    vistos: list[tuple[pd.Timestamp, int]] = []

    def ajustar(self, historico):
        self.ultimo = historico["data"].max() if len(historico) else pd.Timestamp.min
        return self

    def prever(self, jogos):
        assert self.ultimo < jogos["data"].min(), "treinou com jogo do mesmo dia ou do futuro"
        Espiao.vistos.append((jogos["data"].min(), len(jogos)))
        return np.tile([0.5, 0.3, 0.2], (len(jogos), 1))


def test_walk_forward_nunca_treina_com_o_futuro_e_cobre_todos_os_jogos(bruto):
    partidas = limpar(bruto).partidas
    Espiao.vistos = []
    p = walk_forward(partidas, Espiao, temporadas=[2031])
    assert p.shape == ((partidas["temporada"] == 2031).sum(), 3)
    assert sum(n for _, n in Espiao.vistos) == len(p)
    assert len(Espiao.vistos) >= 5  # temporada sintetica dura ~6 semanas: um ajuste por semana
