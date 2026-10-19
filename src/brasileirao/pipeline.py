"""Roda o projeto inteiro, na ordem, e regenera tudo em reports/.

    python -m brasileirao.pipeline            # ~3-4 min
    python -m brasileirao.pipeline --rapido   # pula o tuning e reusa reports/hiperparametros.json
"""

from __future__ import annotations

import argparse
import json
import time

from brasileirao import backtest, dados
from brasileirao.analises import avaliacao, mando, projecao, santos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true", help="reusa os hiperparametros ja escolhidos")
    args = parser.parse_args()

    etapas = []
    if args.rapido:
        def backtest_sem_tuning() -> None:
            partidas = dados.carregar().partidas
            hiper = json.loads(backtest.ARQ_HIPER.read_text(encoding="utf-8"))
            previsoes = backtest.rodar(partidas, hiper)
            previsoes.round(5).to_csv(backtest.ARQ_PREVISOES, index=False, date_format="%Y-%m-%d %H:%M")
            backtest.ARQ_METRICAS.write_text(json.dumps(backtest.avaliar(previsoes), indent=2) + "\n",
                                             encoding="utf-8")
        etapas.append(("backtest (sem tuning)", backtest_sem_tuning))
    else:
        etapas.append(("backtest", backtest.main))
    etapas += [("avaliacao", avaliacao.main), ("mando", mando.main), ("santos", santos.main),
               ("projecao", projecao.main)]

    dados.main([])
    for nome, f in etapas:
        t = time.time()
        print(f"\n=== {nome} ===")
        f()
        print(f"--- {nome}: {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
