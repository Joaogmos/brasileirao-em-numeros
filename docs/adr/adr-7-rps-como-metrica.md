# ADR-7 — RPS e a metrica principal (nao acuracia)

**Status:** aceito
**Data:** 2026-10-19

## Contexto

O produto do modelo e uma probabilidade, nao um palpite. Acuracia so olha o
resultado mais provavel: "34/33/33" e "90/5/5" contam igual se o mandante vence.

## Decisao

Ranked Probability Score (Epstein, 1969) como metrica principal, com log loss e
Brier como secundarias. As tres sao regras de pontuacao **proprias**: o valor
esperado e otimo quando a previsao e a probabilidade verdadeira, entao nao da para
"ganhar" a metrica distorcendo a previsao.

O RPS compara distribuicoes acumuladas, entao respeita a ordem H > D > A: errar
apostando no empate e menos grave que apostar no visitante.
`test_rps_respeita_a_ordem_e_logloss_nao` mostra a diferenca.

Diferencas entre modelos sao testadas com **bootstrap pareado** (os mesmos jogos
nas duas amostras), que separa a variancia da diferenca da variancia de "jogo dificil".

## Consequencias

As diferencas de RPS sao pequenas em valor absoluto (0,2078 x 0,2007), entao todo
numero vem com IC95%. Exemplo: Dixon-Coles - Elo = -0,00001, IC [-0,00114; +0,00113]
— empate tecnico, e e assim que o README apresenta.

## Alternativas consideradas

- **Acuracia:** ignora a confianca da previsao.
- **So log loss:** pune demais previsoes extremas erradas e ignora a ordem.
