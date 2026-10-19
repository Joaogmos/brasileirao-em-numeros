"""Pergunta 2: o rebaixamento do Santos em 2023 foi azar ou desempenho?

Ferramenta: pontos esperados (xPts). Antes de cada jogo, as odds de fechamento
dizem a chance de vitoria/empate/derrota; xPts = 3*P(vitoria) + 1*P(empate).
Somando os 38 jogos, xPts e "quantos pontos se esperava, rodada a rodada, com toda
a informacao disponivel antes da bola rolar". Pontos - xPts = sorte/azar + o que o
mercado nao conseguiu antecipar.

Atencao ao que isso NAO mede: o mercado ja sabia que o Santos estava mal (o xPts
de cada jogo ja embute a fase ruim). Logo "azar" aqui e so a parte que nem quem
acompanhava o time de perto esperava.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brasileirao import config, estilo, odds
from brasileirao.backtest import ARQ_HIPER
from brasileirao.dados import carregar
from brasileirao.modelos.dixon_coles import DixonColes
from brasileirao.simulacao import jogos_restantes, simular
from brasileirao.tabela import classificacao

CLUBE = "Santos"
ARQ = config.RELATORIOS / "santos.json"
Z4 = 17  # da 17a posicao para baixo = rebaixado


def xpts(partidas: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Pontos reais e esperados por clube e temporada."""
    p = partidas.reset_index(drop=True)
    pts_casa = np.select([p["gols_mandante"] > p["gols_visitante"],
                          p["gols_mandante"] == p["gols_visitante"]], [3, 1], 0)
    casa = pd.DataFrame({"temporada": p["temporada"], "clube": p["mandante"], "local": "casa",
                         "pontos": pts_casa, "xpts": 3 * probs[:, 0] + probs[:, 1]})
    fora = pd.DataFrame({"temporada": p["temporada"], "clube": p["visitante"], "local": "fora",
                         "pontos": np.select([pts_casa == 0, pts_casa == 1], [3, 1], 0),
                         "xpts": 3 * probs[:, 2] + probs[:, 1]})
    return pd.concat([casa, fora], ignore_index=True)


def resumo_xpts(partidas: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    r = xpts(partidas, probs).groupby(["temporada", "clube"])[["pontos", "xpts"]].sum().reset_index()
    r["diferenca"] = r["pontos"] - r["xpts"]
    return r


def simular_pontos(partidas: pd.DataFrame, probs: np.ndarray, n: int = 20000,
                   semente: int = config.SEMENTE) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Rejoga a temporada `n` vezes sorteando cada resultado com as probabilidades do mercado.

    Devolve (clubes, pontos[n, clubes], posicao[n, clubes]). Empate em pontos e
    desfeito por vitorias e depois por sorteio (sem placar nao ha saldo).
    """
    rng = np.random.default_rng(semente)
    clubes = sorted(set(partidas["mandante"]) | set(partidas["visitante"]))
    idx = {c: i for i, c in enumerate(clubes)}
    ic = partidas["mandante"].map(idx).to_numpy()
    ifo = partidas["visitante"].map(idx).to_numpy()

    u = rng.random((n, len(partidas)))
    cum = np.cumsum(probs, axis=1)
    res = (u[:, :, None] > cum[None, :, :-1]).sum(axis=2)  # 0=H, 1=D, 2=A

    pontos = np.zeros((n, len(clubes)))
    vitorias = np.zeros((n, len(clubes)))
    for j in range(len(partidas)):
        r = res[:, j]
        pontos[:, ic[j]] += np.select([r == 0, r == 1], [3, 1], 0)
        pontos[:, ifo[j]] += np.select([r == 2, r == 1], [3, 1], 0)
        vitorias[:, ic[j]] += r == 0
        vitorias[:, ifo[j]] += r == 2

    chave = pontos * 1e4 + vitorias * 1e2 + rng.random(pontos.shape)
    posicao = (-chave).argsort(axis=1).argsort(axis=1) + 1
    return clubes, pontos, posicao


def pre_temporada(partidas: pd.DataFrame, temporada: int = 2023) -> dict:
    """O que o Dixon-Coles esperava ANTES da 1a rodada (so com dados ate a temporada anterior)."""
    params = json.loads(ARQ_HIPER.read_text(encoding="utf-8"))["dixon_coles"]["melhor"]
    alvo = partidas[partidas["temporada"] == temporada]
    modelo = DixonColes(**params).ajustar(partidas[partidas["data"] < alvo["data"].min()])
    calendario = jogos_restantes(alvo.iloc[:0], sorted(set(alvo["mandante"])))
    sim = simular(modelo, alvo.iloc[:0], calendario, n=20000)
    dist = sim["posicoes"].loc[CLUBE]
    return {
        "p_rebaixamento": float(dist.loc[Z4:].sum()),
        "posicao_media": float(sim["resumo"].loc[CLUBE, "posicao_media"]),
        "pontos_esperados": float(sim["resumo"].loc[CLUBE, "pontos_esperados"]),
        "ranking_forca_pre_temporada": int(
            sim["resumo"]["pontos_esperados"].rank(ascending=False).loc[CLUBE]),
    }


def analisar_2023(partidas: pd.DataFrame) -> dict:
    temporada = partidas[partidas["temporada"] == 2023].reset_index(drop=True)
    probs = odds.probabilidades_mercado(temporada)
    clubes, pontos, posicao = simular_pontos(temporada, probs)
    i = clubes.index(CLUBE)
    real = classificacao(temporada).set_index("clube")

    # Linha de corte: pontos do 17o colocado em cada simulacao
    corte = np.sort(pontos, axis=1)[:, len(clubes) - Z4]

    detalhe = xpts(temporada, probs)
    detalhe = detalhe[detalhe["clube"] == CLUBE].assign(
        turno=lambda d: np.where(d.index % len(temporada) < len(temporada) / 2, "1o turno", "2o turno"))
    jogos = temporada.assign(santos_casa=temporada["mandante"] == CLUBE)
    jogos = jogos[(jogos["mandante"] == CLUBE) | (jogos["visitante"] == CLUBE)]
    p_santos = odds.probabilidades_mercado(jogos)
    p_vitoria = np.where(jogos["santos_casa"], p_santos[:, 0], p_santos[:, 2])
    saldo = np.where(jogos["santos_casa"], 1, -1) * (jogos["gols_mandante"] - jogos["gols_visitante"])

    return {
        "pontos_reais": int(real.loc[CLUBE, "pontos"]),
        "posicao_real": int(real.loc[CLUBE, "posicao"]),
        "xpts_mercado": float(detalhe["xpts"].sum()),
        "pontos_simulados_media": float(pontos[:, i].mean()),
        "pontos_simulados_ic90": [float(x) for x in np.quantile(pontos[:, i], [0.05, 0.95])],
        "p_pontos_ate_o_real": float((pontos[:, i] <= real.loc[CLUBE, "pontos"]).mean()),
        "p_rebaixamento_dadas_as_odds": float((posicao[:, i] >= Z4).mean()),
        "posicao_simulada_mediana": float(np.median(posicao[:, i])),
        "linha_de_corte_simulada_media": float(corte.mean()),
        "pontos_por_local": detalhe.groupby("local")[["pontos", "xpts"]].sum().round(2).to_dict("index"),
        "pontos_por_turno": detalhe.groupby("turno")[["pontos", "xpts"]].sum().round(2).to_dict("index"),
        "jogos_como_favorito": int((p_vitoria > 0.5).sum()),
        "derrotas_como_favorito": int(((p_vitoria > 0.5) & (saldo < 0)).sum()),
        "gols_sofridos": int(real.loc[CLUBE, "gols_contra"]),
        "goleadas_sofridas_3_ou_mais": int((saldo <= -3).sum()),
        "pre_temporada_dixon_coles": pre_temporada(partidas, 2023),
        "_pontos_simulados": pontos[:, i],
    }


def figura_historico(r: pd.DataFrame, destino=config.FIGURAS / "santos_pontos_vs_xpts.png") -> None:
    estilo.aplicar()
    s = r[r["clube"] == CLUBE].set_index("temporada").reindex(range(2012, int(r["temporada"].max()) + 1))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    anos = s.index.to_numpy()
    largura = 0.38
    ax.bar(anos - largura / 2, s["xpts"], largura, color=estilo.CINZA_CLARO, label="Pontos esperados (mercado)")
    cores = [estilo.VERMELHO if d < -5 else estilo.SANTOS for d in s["diferenca"].fillna(0)]
    ax.bar(anos + largura / 2, s["pontos"], largura, color=cores, label="Pontos reais")
    for ano, linha in s.iterrows():
        if pd.isna(linha["pontos"]):
            ax.text(ano, 3, "Serie B", ha="center", va="bottom", rotation=90, color=estilo.MUDO, fontsize=9)
        elif abs(linha["diferenca"]) >= 5:
            ax.text(ano + largura / 2, linha["pontos"] + 1, f"{linha['diferenca']:+.0f}", ha="center",
                    va="bottom", fontsize=9, color=estilo.TINTA_2)
    ax.set_xticks(anos)
    ax.set_title("Santos: pontos reais x pontos esperados pelas odds, por temporada")
    ax.set_ylabel("pontos")
    ax.legend(loc="upper right", fontsize=9, ncols=2)
    ax.text(0, -0.16, "2026 em andamento. Vermelho: >5 pontos abaixo do esperado.", transform=ax.transAxes,
            fontsize=8.5, color=estilo.MUDO)
    ax.set_ylim(0, max(s["pontos"].max(), s["xpts"].max()) + 12)
    fig.savefig(destino)
    plt.close(fig)


def figura_2023(a: dict, destino=config.FIGURAS / "santos_2023_simulacao.png") -> None:
    estilo.aplicar()
    pts = a["_pontos_simulados"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    valores, contagem = np.unique(pts, return_counts=True)
    cores = [estilo.VERMELHO if v <= a["pontos_reais"] else estilo.CINZA_CLARO for v in valores]
    ax.bar(valores, contagem / len(pts) * 100, width=0.85, color=cores)
    ax.set_ylim(0, contagem.max() / len(pts) * 100 * 1.45)
    ax.axvline(a["pontos_reais"], color=estilo.TINTA, lw=1.5)
    ax.text(a["pontos_reais"] - 0.6, ax.get_ylim()[1] * 0.92,
            f"real: {a['pontos_reais']} pts\n{a['p_pontos_ate_o_real']:.0%} das simulacoes\nfizeram isso ou menos",
            ha="right", va="top", fontsize=9)
    ax.axvline(a["xpts_mercado"], color=estilo.MUDO, lw=1, ls="--")
    ax.text(a["xpts_mercado"] + 0.6, ax.get_ylim()[1] * 0.92, f"esperado: {a['xpts_mercado']:.1f}",
            fontsize=9, va="top", color=estilo.TINTA_2)
    ax.set_title("Santos 2023: pontos em 20 mil temporadas simuladas com as odds de cada jogo")
    ax.set_xlabel("pontos ao fim das 38 rodadas")
    ax.set_ylabel("% das simulacoes")
    fig.savefig(destino)
    plt.close(fig)


def main() -> None:
    partidas = carregar().partidas
    probs = odds.probabilidades_mercado(partidas)
    r = resumo_xpts(partidas, probs)
    a = analisar_2023(partidas)

    encerradas = r[r["temporada"] < r["temporada"].max()].sort_values("diferenca")
    rank = int((encerradas["diferenca"] < encerradas.query("clube == @CLUBE and temporada == 2023")
                ["diferenca"].iloc[0]).sum()) + 1

    figura_historico(r)
    figura_2023(a)

    saida = {k: v for k, v in a.items() if not k.startswith("_")}
    saida["ranking_azar_entre_todas_as_campanhas"] = {"posicao": rank, "de": len(encerradas)}
    saida["piores_campanhas_vs_esperado"] = encerradas.head(8).round(2).to_dict("records")
    saida["historico_santos"] = r[r["clube"] == CLUBE].round(2).to_dict("records")
    ARQ.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(saida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
