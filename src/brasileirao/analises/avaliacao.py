"""Pergunta 3: da para prever o Brasileirao melhor do que o mercado?

Le as previsoes fora da amostra geradas pelo backtest e desenha as tres figuras
que respondem a pergunta: comparacao de RPS, calibracao e estabilidade por temporada.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brasileirao import config, estilo, metricas
from brasileirao.backtest import ARQ_METRICAS, ARQ_PREVISOES, probs

ROTULOS = {
    "frequencia": "Baseline (frequencia)",
    "elo": "Elo",
    "dixon_coles": "Dixon-Coles",
    "media_dc_elo": "Media DC + Elo",
    "mercado": "Mercado (odds)",
}


def figura_rps(m: dict, destino=config.FIGURAS / "rps_modelos.png") -> None:
    estilo.aplicar()
    nomes = list(ROTULOS)
    valores = [m["geral"][n]["rps"] for n in nomes]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    cores = [estilo.LARANJA if n == "mercado" else estilo.AZUL if n != "frequencia" else estilo.CINZA_CLARO
             for n in nomes]
    y = np.arange(len(nomes))[::-1]
    ax.barh(y, valores, color=cores, height=0.6)
    for yi, v in zip(y, valores, strict=True):
        ax.text(v + 0.0004, yi, f"{v:.4f}", va="center", fontsize=9, color=estilo.TINTA_2)
    ax.set_yticks(y, [ROTULOS[n] for n in nomes])
    ax.set_xlim(0.195, max(valores) + 0.003)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"RPS medio fora da amostra, {m['temporadas'][0]}-{m['temporadas'][1]} "
                 f"({m['n_jogos']} jogos) - menor e melhor")
    ax.text(0, -0.2, "Eixo comeca em 0,195 para mostrar as diferencas; elas sao pequenas em valor absoluto, "
            "mas o IC95% do bootstrap pareado e o que diz se sao reais.", transform=ax.transAxes,
            fontsize=8, color=estilo.MUDO)
    fig.savefig(destino)
    plt.close(fig)


def figura_calibracao(prev: pd.DataFrame, destino=config.FIGURAS / "calibracao.png") -> None:
    estilo.aplicar()
    y = metricas.indices(prev["resultado"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 0.8], [0, 0.8], color=estilo.MUDO, lw=1, ls="--")
    ax.text(0.62, 0.66, "calibracao\nperfeita", fontsize=8.5, color=estilo.MUDO, rotation=0)
    for nome, cor in (("dixon_coles", estilo.AZUL), ("mercado", estilo.LARANJA)):
        c = metricas.calibracao(probs(prev, nome), y, n_faixas=10)
        c = c[c["n"] >= 30]
        ax.plot(c["prevista"], c["observada"], color=cor, marker="o", ms=5, label=ROTULOS[nome])
    ax.set_xlabel("probabilidade prevista")
    ax.set_ylabel("frequencia observada")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.set_title("Calibracao (faixas com 30+ previsoes)")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(destino)
    plt.close(fig)


def figura_por_temporada(m: dict, destino=config.FIGURAS / "rps_por_temporada.png") -> None:
    estilo.aplicar()
    t = pd.DataFrame(m["por_temporada"]).T
    t.index = t.index.astype(int)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for nome, cor in (("frequencia", estilo.CINZA_CLARO), ("dixon_coles", estilo.AZUL),
                      ("mercado", estilo.LARANJA)):
        ax.plot(t.index, t[nome], color=cor, marker="o", ms=4, label=ROTULOS[nome])
    ax.set_xticks(t.index)
    ax.set_title("RPS por temporada: o ranking dos modelos se mantem ano a ano?")
    ax.set_ylim(0.17, 0.242)
    ax.legend(fontsize=9, ncols=3, loc="upper right")
    fig.savefig(destino)
    plt.close(fig)


def resumo_textual(m: dict) -> dict:
    g = m["geral"]
    ganho_mercado = g["frequencia"]["rps"] - g["mercado"]["rps"]
    ganho_dc = g["frequencia"]["rps"] - g["dixon_coles"]["rps"]
    vitorias_temporada = sum(v["dixon_coles"] < v["mercado"] for v in m["por_temporada"].values())
    return {
        "fracao_do_ganho_do_mercado_capturada_pelo_dc": ganho_dc / ganho_mercado,
        "temporadas_em_que_dc_bateu_o_mercado": vitorias_temporada,
        "temporadas_avaliadas": len(m["por_temporada"]),
    }


def main() -> None:
    m = json.loads(ARQ_METRICAS.read_text(encoding="utf-8"))
    prev = pd.read_csv(ARQ_PREVISOES, parse_dates=["data"])
    figura_rps(m)
    figura_calibracao(prev)
    figura_por_temporada(m)
    print(json.dumps(resumo_textual(m), indent=2))


if __name__ == "__main__":
    main()
