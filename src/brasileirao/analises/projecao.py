"""Pergunta 4: onde o Santos termina o Brasileirao em andamento?

Ajusta o Dixon-Coles (hiperparametros da validacao) com todos os jogos ate a
ultima rodada do snapshot, deduz os jogos que faltam e simula o resto da
temporada 20 mil vezes.

Zonas: "top 6" aproxima as vagas diretas na Libertadores (o numero exato depende
de quem vence a Copa do Brasil e a propria Libertadores); Z4 = 17o a 20o.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from brasileirao import config, estilo
from brasileirao.analises.santos import CLUBE
from brasileirao.backtest import ARQ_HIPER
from brasileirao.dados import carregar
from brasileirao.modelos.dixon_coles import DixonColes
from brasileirao.simulacao import jogos_restantes, simular, zonas
from brasileirao.tabela import classificacao

N_SIMULACOES = 20000
ARQ_JSON = config.RELATORIOS / "projecao.json"
ARQ_CSV = config.RELATORIOS / "projecao_posicoes.csv"


def projetar(partidas: pd.DataFrame, temporada: int | None = None, n: int = N_SIMULACOES) -> dict:
    temporada = temporada or int(partidas["temporada"].max())
    params = json.loads(ARQ_HIPER.read_text(encoding="utf-8"))["dixon_coles"]["melhor"]
    jogados = partidas[partidas["temporada"] == temporada]
    modelo = DixonColes(**params).ajustar(partidas[partidas["data"] <= jogados["data"].max()])
    restantes = jogos_restantes(jogados)
    sim = simular(modelo, jogados, restantes, n=n)
    sim.update(temporada=temporada, jogados=jogados, restantes=restantes, modelo=modelo,
               ultima_data=jogados["data"].max())
    return sim


def validar_no_passado(partidas: pd.DataFrame, temporadas=range(2017, 2026), jogos_disputados: int = 300,
                       n: int = 5000) -> dict:
    """A projecao funciona? Refaz o exercicio "com 300 jogos disputados" nas temporadas
    encerradas e compara a probabilidade dada com o que aconteceu de fato.
    """
    params = json.loads(ARQ_HIPER.read_text(encoding="utf-8"))["dixon_coles"]["melhor"]
    linhas = []
    for t in temporadas:
        temporada = partidas[partidas["temporada"] == t].sort_values("data")
        jogados = temporada.iloc[:jogos_disputados]
        modelo = DixonColes(**params).ajustar(partidas[partidas["data"] <= jogados["data"].max()])
        sim = simular(modelo, jogados, jogos_restantes(jogados, sorted(set(temporada["mandante"]))), n=n)
        z = zonas(sim["posicoes"])
        final = classificacao(temporada).set_index("clube")["posicao"]
        for clube, linha in z.iterrows():
            linhas.append({"temporada": t, "clube": clube, "p_rebaixamento": linha["rebaixamento"],
                           "p_top6": linha["top6"], "p_titulo": linha["titulo"],
                           "rebaixado": int(final[clube] >= 17), "top6": int(final[clube] <= 6),
                           "campeao": int(final[clube] == 1)})
    v = pd.DataFrame(linhas)
    resultado = {"temporadas": [min(temporadas), max(temporadas)], "jogos_disputados": jogos_disputados}
    for evento, prob in (("rebaixado", "p_rebaixamento"), ("top6", "p_top6"), ("campeao", "p_titulo")):
        base = v[evento].mean()
        brier = float(((v[prob] - v[evento]) ** 2).mean())
        resultado[evento] = {
            "brier": brier,
            "brier_ingenuo": float(((base - v[evento]) ** 2).mean()),
            "esperados": float(v[prob].sum()),
            "observados": int(v[evento].sum()),
            "acima_de_90pct": {"casos": int((v[prob] >= 0.9).sum()),
                               "aconteceu": int(v.loc[v[prob] >= 0.9, evento].sum())},
            "entre_10_e_90pct": {"casos": int(v[prob].between(0.1, 0.9).sum()),
                                 "prob_media": float(v.loc[v[prob].between(0.1, 0.9), prob].mean()),
                                 "taxa_real": float(v.loc[v[prob].between(0.1, 0.9), evento].mean())},
        }
    return resultado


def figura_heatmap(sim: dict, destino=config.FIGURAS / "projecao_heatmap.png") -> None:
    estilo.aplicar()
    dist = sim["posicoes"]
    ordem = sim["resumo"]["posicao_media"].sort_values().index
    d = dist.loc[ordem] * 100
    cmap = LinearSegmentedColormap.from_list("azul", estilo.AZUL_SEQ)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(d.to_numpy(), cmap=cmap, vmin=0, vmax=60, aspect="auto")
    for i, clube in enumerate(d.index):
        for j, v in enumerate(d.loc[clube]):
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 30 else estilo.TINTA_2)
    ax.set_xticks(range(d.shape[1]), d.columns)
    ax.set_yticks(range(len(d)), [f"{c}  ({int(sim['resumo'].loc[c, 'pontos_atuais'])} pts)" for c in d.index])
    for rotulo in ax.get_yticklabels():
        if rotulo.get_text().startswith(CLUBE):
            rotulo.set_fontweight("bold")
            rotulo.set_color(estilo.TINTA)
    i_santos = list(d.index).index(CLUBE)
    ax.add_patch(plt.Rectangle((-0.5, i_santos - 0.5), d.shape[1], 1, fill=False, ec=estilo.TINTA, lw=1.5))
    ax.axvline(5.5, color=estilo.LARANJA, lw=1.2)  # coluna j = posicao j+1: entre a 6a e a 7a
    ax.axvline(15.5, color=estilo.VERMELHO, lw=1.2)  # entre a 16a e a 17a
    ax.grid(False)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.set_title(f"Brasileirao {sim['temporada']}: chance (%) de terminar em cada posicao - "
                 f"{len(sim['restantes'])} jogos restantes, {N_SIMULACOES // 1000} mil simulacoes", pad=28)
    ax.text(0, -0.04, f"Dados ate {sim['ultima_data']:%d/%m/%Y}. Linha laranja: fim do top 6; "
            "linha vermelha: inicio do Z4. Celulas < 0,5% em branco.",
            transform=ax.transAxes, fontsize=8.5, color=estilo.MUDO, va="top")
    fig.savefig(destino)
    plt.close(fig)


def figura_santos(sim: dict, destino=config.FIGURAS / "projecao_santos.png") -> None:
    estilo.aplicar()
    dist = sim["posicoes"].loc[CLUBE] * 100
    fig, ax = plt.subplots(figsize=(8, 3.2))
    cores = [estilo.LARANJA if p <= 6 else estilo.VERMELHO if p >= 17 else estilo.SANTOS for p in dist.index]
    ax.bar(dist.index, dist.to_numpy(), color=cores, width=0.8)
    for p, v in dist.items():
        if v >= 1:
            ax.text(p, v + 0.4, f"{v:.0f}%", ha="center", fontsize=8.5, color=estilo.TINTA_2)
    ax.set_xticks(dist.index)
    ax.set_xlabel("posicao final")
    ax.set_ylabel("% das simulacoes")
    r = sim["resumo"].loc[CLUBE]
    ax.set_title(f"Santos {sim['temporada']}: posicao final projetada "
                 f"(hoje {int(r['pontos_atuais'])} pts; esperado {r['pontos_esperados']:.0f} pts)")
    fig.savefig(destino)
    plt.close(fig)


def main() -> None:
    partidas = carregar().partidas
    sim = projetar(partidas)
    z = zonas(sim["posicoes"])
    tabela = (classificacao(sim["jogados"]).set_index("clube")[["posicao", "pontos", "jogos"]]
              .join(sim["resumo"][["pontos_esperados", "pontos_p05", "pontos_p95", "posicao_media"]])
              .join(z).sort_values("posicao_media"))

    figura_heatmap(sim)
    figura_santos(sim)
    sim["posicoes"].round(5).to_csv(ARQ_CSV, index_label="clube")

    santos = tabela.loc[CLUBE]
    validacao = validar_no_passado(partidas)
    saida = {
        "validacao_no_passado": validacao,
        "temporada": sim["temporada"],
        "dados_ate": f"{sim['ultima_data']:%Y-%m-%d}",
        "jogos_restantes": len(sim["restantes"]),
        "simulacoes": N_SIMULACOES,
        "santos": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in santos.items()},
        "tabela_projetada": tabela.round(4).reset_index().to_dict("records"),
    }
    ARQ_JSON.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(tabela.round(3).to_string())
    print(json.dumps(validacao, indent=2))


if __name__ == "__main__":
    main()
