"""Validacao walk-forward: o modelo so ve o passado, como veria na vida real.

Para cada semana do periodo de teste:
    1. ajusta o modelo com TODOS os jogos anteriores ao inicio da semana;
    2. preve os jogos da semana;
    3. avanca uma semana.

Validacao cruzada k-fold aleatoria seria vazamento: treinar com a rodada 30 para
prever a rodada 10 usa informacao do futuro (forca do elenco, forma). Ver ADR-5.

Divisao temporal (nada se sobrepoe):
    2012-2013  aquecimento (so treino)
    2014-2016  validacao: escolha de hiperparametros
    2017-2026  teste: numeros reportados no README, tocados uma unica vez
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from brasileirao import config, metricas, odds
from brasileirao.dados import carregar
from brasileirao.modelos.dixon_coles import DixonColes
from brasileirao.modelos.elo import Elo
from brasileirao.modelos.frequencia import Frequencia

VALIDACAO = (2014, 2015, 2016)
TESTE = tuple(range(2017, 2027))

GRADE_DC = {"xi": [0.001, 0.002, 0.003, 0.005], "reg": [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]}
GRADE_ELO = {"k": [3.0, 5.0, 10.0, 15.0, 20.0], "mando": [0.0, 25.0, 50.0, 75.0], "regressao": [0.0, 0.1, 0.25]}

ARQ_HIPER = config.RELATORIOS / "hiperparametros.json"
ARQ_PREVISOES = config.RELATORIOS / "previsoes_teste.csv"
ARQ_METRICAS = config.RELATORIOS / "metricas.json"


def walk_forward(partidas: pd.DataFrame, fabrica: Callable[[], object], temporadas) -> np.ndarray:
    """Previsoes fora da amostra para todos os jogos das `temporadas`, semana a semana."""
    alvo = partidas["temporada"].isin(temporadas).to_numpy()
    semana = partidas["data"].dt.to_period("W").astype(str).to_numpy()
    previsoes = np.full((len(partidas), 3), np.nan)
    modelo = fabrica()  # mesma instancia: o Dixon-Coles aproveita o ajuste anterior (warm start)
    for s in pd.unique(semana[alvo]):
        bloco = alvo & (semana == s)
        inicio = partidas.loc[bloco, "data"].min()
        modelo.ajustar(partidas[partidas["data"] < inicio])
        previsoes[bloco] = modelo.prever(partidas[bloco])
    return previsoes[alvo]


def _rps_medio(partidas: pd.DataFrame, fabrica, temporadas) -> float:
    p = walk_forward(partidas, fabrica, temporadas)
    y = metricas.indices(partidas.loc[partidas["temporada"].isin(temporadas), "resultado"])
    return float(metricas.rps(p, y).mean())


def tunar(partidas: pd.DataFrame) -> dict:
    """Busca em grade, SO nas temporadas de validacao."""
    base = partidas[partidas["temporada"] <= max(VALIDACAO)]
    resultado = {}
    for nome, classe, grade in (("dixon_coles", DixonColes, GRADE_DC), ("elo", Elo, GRADE_ELO)):
        tentativas = []
        for valores in itertools.product(*grade.values()):
            params = dict(zip(grade, valores, strict=True))
            rps = _rps_medio(base, lambda c=classe, p=params: c(**p), VALIDACAO)
            tentativas.append({**params, "rps": rps})
        melhor = min(tentativas, key=lambda t: t["rps"])
        resultado[nome] = {"melhor": {k: v for k, v in melhor.items() if k != "rps"},
                           "rps_validacao": melhor["rps"], "grade": tentativas}
        print(f"  {nome}: {resultado[nome]['melhor']}  (RPS validacao {melhor['rps']:.4f})")
    return resultado


def rodar(partidas: pd.DataFrame, hiper: dict) -> pd.DataFrame:
    """Previsoes no periodo de teste de todos os modelos + o mercado, lado a lado."""
    teste = partidas[partidas["temporada"].isin(TESTE)].reset_index(drop=True)
    dc = hiper["dixon_coles"]["melhor"]
    el = hiper["elo"]["melhor"]
    previsoes = {
        "frequencia": walk_forward(partidas, Frequencia, TESTE),
        "elo": walk_forward(partidas, lambda: Elo(**el), TESTE),
        "dixon_coles": walk_forward(partidas, lambda: DixonColes(**dc), TESTE),
    }
    previsoes["media_dc_elo"] = (previsoes["dixon_coles"] + previsoes["elo"]) / 2
    previsoes["mercado"] = odds.probabilidades_mercado(teste, "proporcional")
    previsoes["mercado_shin"] = odds.probabilidades_mercado(teste, "shin")

    saida = teste[["temporada", "data", "mandante", "visitante", "gols_mandante", "gols_visitante",
                   "resultado"]].copy()
    for nome, p in previsoes.items():
        for j, r in enumerate(config.RESULTADOS):
            saida[f"{nome}_{r}"] = p[:, j]
    return saida


MODELOS = ["frequencia", "elo", "dixon_coles", "media_dc_elo", "mercado", "mercado_shin"]


def probs(previsoes: pd.DataFrame, modelo: str) -> np.ndarray:
    return previsoes[[f"{modelo}_{r}" for r in config.RESULTADOS]].to_numpy()


def _arredondar(obj, casas: int = 5):
    """Arredonda floats aninhados: o JSON versionado nao muda por ruido na 10a casa do otimizador."""
    if isinstance(obj, float):
        return round(obj, casas)
    if isinstance(obj, dict):
        return {k: _arredondar(v, casas) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_arredondar(v, casas) for v in obj]
    return obj


def avaliar(previsoes: pd.DataFrame) -> dict:
    y = metricas.indices(previsoes["resultado"])
    geral = {m: metricas.resumo(probs(previsoes, m), y) for m in MODELOS}

    por_temporada = {}
    for t, g in previsoes.groupby("temporada"):
        yt = metricas.indices(g["resultado"])
        por_temporada[int(t)] = {m: float(metricas.rps(probs(g, m), yt).mean()) for m in MODELOS}

    rps = {m: metricas.rps(probs(previsoes, m), y) for m in MODELOS}
    comparacoes = {
        "dixon_coles - frequencia": metricas.bootstrap_diferenca(rps["dixon_coles"], rps["frequencia"]),
        "dixon_coles - elo": metricas.bootstrap_diferenca(rps["dixon_coles"], rps["elo"]),
        "media_dc_elo - dixon_coles": metricas.bootstrap_diferenca(rps["media_dc_elo"], rps["dixon_coles"]),
        "dixon_coles - mercado": metricas.bootstrap_diferenca(rps["dixon_coles"], rps["mercado"]),
        "mercado_shin - mercado": metricas.bootstrap_diferenca(rps["mercado_shin"], rps["mercado"]),
    }
    return _arredondar({"n_jogos": len(previsoes), "temporadas": [min(TESTE), max(TESTE)], "geral": geral,
                        "por_temporada": por_temporada, "comparacoes_rps": comparacoes})


def main() -> None:
    partidas = carregar().partidas
    config.RELATORIOS.mkdir(exist_ok=True)

    t0 = time.time()
    print("tunando hiperparametros em", VALIDACAO, "...")
    hiper = tunar(partidas)
    ARQ_HIPER.write_text(json.dumps(_arredondar(hiper), indent=2) + "\n", encoding="utf-8")

    print("walk-forward no teste", TESTE[0], "-", TESTE[-1], "...")
    previsoes = rodar(partidas, hiper)
    previsoes.round(5).to_csv(ARQ_PREVISOES, index=False, date_format="%Y-%m-%d %H:%M")

    resultado = avaliar(previsoes)
    ARQ_METRICAS.write_text(json.dumps(resultado, indent=2) + "\n", encoding="utf-8")

    print(f"\n{resultado['n_jogos']} jogos de teste ({time.time() - t0:.0f}s)\n")
    tabela = pd.DataFrame(resultado["geral"]).T
    print(tabela.round(4).to_string())
    print()
    for nome, c in resultado["comparacoes_rps"].items():
        print(f"  {nome:28s} {c['diferenca']:+.5f}  IC95% [{c['ic_inf']:+.5f}, {c['ic_sup']:+.5f}]")


if __name__ == "__main__":
    main()
