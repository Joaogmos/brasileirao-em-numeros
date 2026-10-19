"""Pergunta 1: quanto vale jogar em casa no Brasileirao -- e o que os portoes
fechados de 2020 mostram sobre isso?

A temporada 2020 (ago/2020 a fev/2021) foi disputada inteira sem publico. E um
"experimento natural": a torcida sumiu e o resto (viagem, gramado, rotina) ficou.
A temporada 2021 so recebeu publico a partir de outubro, entao entra como mista
e fica fora das duas comparacoes.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brasileirao import config, estilo, odds
from brasileirao.dados import carregar
from brasileirao.modelos.dixon_coles import DixonColes

COM_PUBLICO = tuple(range(2012, 2020))
SEM_PUBLICO = (2020,)
ARQ = config.RELATORIOS / "mando.json"


def _preparar(partidas: pd.DataFrame) -> pd.DataFrame:
    p = partidas.copy()
    p["saldo"] = p["gols_mandante"] - p["gols_visitante"]
    p["pontos_mandante"] = np.select([p["saldo"] > 0, p["saldo"] == 0], [3, 1], 0)
    p["mercado_p_mandante"] = odds.probabilidades_mercado(p)[:, 0]
    return p


def por_temporada(partidas: pd.DataFrame) -> pd.DataFrame:
    p = _preparar(partidas)
    tabela = p.groupby("temporada").agg(
        jogos=("saldo", "size"),
        vitoria_mandante=("resultado", lambda r: (r == "H").mean()),
        empate=("resultado", lambda r: (r == "D").mean()),
        vitoria_visitante=("resultado", lambda r: (r == "A").mean()),
        saldo_medio=("saldo", "mean"),
        pontos_por_jogo_mandante=("pontos_mandante", "mean"),
        mercado_p_mandante=("mercado_p_mandante", "mean"),
    )
    # Vantagem de mando "limpa" da forca dos times: parametro do Dixon-Coles ajustado
    # so com a temporada (sem peso temporal). exp(mando) = multiplicador dos gols em casa.
    tabela["multiplicador_gols_casa"] = [
        float(np.exp(DixonColes(xi=0.0, reg=8.0).ajustar(g).mando)) for _, g in p.groupby("temporada")
    ]
    return tabela


def portoes_fechados(partidas: pd.DataFrame, n_boot: int = 10000, semente: int = config.SEMENTE) -> dict:
    p = _preparar(partidas)
    com = p.loc[p["temporada"].isin(COM_PUBLICO), "saldo"].to_numpy()
    sem = p.loc[p["temporada"].isin(SEM_PUBLICO), "saldo"].to_numpy()

    rng = np.random.default_rng(semente)
    boot = (rng.choice(sem, (n_boot, len(sem))).mean(axis=1)
            - rng.choice(com, (n_boot, len(com))).mean(axis=1))

    # Teste de permutacao: se o publico nao importasse, rotular 380 jogos quaisquer
    # como "sem publico" daria uma queda tao grande quanto a observada com que frequencia?
    juntos = np.concatenate([com, sem])
    obs = sem.mean() - com.mean()
    perm = np.array([
        (lambda x: x[: len(sem)].mean() - x[len(sem):].mean())(rng.permutation(juntos))
        for _ in range(n_boot)
    ])

    # A pergunta que o teste por jogo NAO responde: a variacao normal ENTRE temporadas
    saldo_temp = p.groupby("temporada")["saldo"].mean()
    saldos_com = saldo_temp.loc[list(COM_PUBLICO)]
    return {
        "saldo_com_publico": float(com.mean()),
        "saldo_sem_publico": float(sem.mean()),
        "diferenca": float(obs),
        "ic95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "p_valor_permutacao": float((perm <= obs).mean()),
        "queda_relativa": float(obs / com.mean()),
        "temporadas_com_publico_tao_baixas_quanto_2020": [
            int(t) for t, s in saldos_com.items() if s <= saldo_temp[2020] + 0.01
        ],
        "desvio_padrao_entre_temporadas_com_publico": float(saldos_com.std(ddof=1)),
        "saldo_2022_2026": float(p.loc[p["temporada"] >= 2022, "saldo"].mean()),
    }


def figura(tabela: pd.DataFrame, destino=config.FIGURAS / "mando_por_temporada.png") -> None:
    estilo.aplicar()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    anos = tabela.index.to_numpy()

    a1.axvspan(2019.5, 2020.5, color=estilo.CINZA_CLARO, alpha=0.5, lw=0)
    a1.text(2020, tabela["saldo_medio"].max() + 0.02, "sem\npublico", ha="center", va="bottom",
            color=estilo.TINTA_2, fontsize=9)
    a1.plot(anos, tabela["saldo_medio"], color=estilo.AZUL, marker="o", ms=5)
    media_com = tabela.loc[list(COM_PUBLICO), "saldo_medio"].mean()
    a1.axhline(media_com, color=estilo.MUDO, lw=1, ls="--")
    a1.text(anos[0], media_com + 0.01, f"media 2012-19: {media_com:+.2f}", color=estilo.TINTA_2,
            fontsize=9, va="bottom")
    a1.set_title("Saldo medio de gols do mandante por jogo")
    a1.set_ylim(0, tabela["saldo_medio"].max() + 0.15)
    a1.set_xticks(anos[::2])

    a2.axvspan(2019.5, 2020.5, color=estilo.CINZA_CLARO, alpha=0.5, lw=0)
    a2.plot(anos, tabela["vitoria_mandante"] * 100, color=estilo.AZUL, marker="o", ms=5, label="Real")
    a2.plot(anos, tabela["mercado_p_mandante"] * 100, color=estilo.LARANJA, marker="o", ms=5,
            label="Prevista pelo mercado (odds)")
    a2.set_title("Vitorias do mandante (%)")
    a2.legend(loc="lower left", fontsize=9)
    a2.set_xticks(anos[::2])
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino)
    plt.close(fig)


def main() -> None:
    partidas = carregar().partidas
    tabela = por_temporada(partidas)
    teste = portoes_fechados(partidas)
    figura(tabela)
    ARQ.write_text(json.dumps({"portoes_fechados": teste,
                               "por_temporada": tabela.round(4).reset_index().to_dict("records")},
                              indent=2) + "\n", encoding="utf-8")
    print(tabela.round(3).to_string())
    print(json.dumps(teste, indent=2))


if __name__ == "__main__":
    main()
