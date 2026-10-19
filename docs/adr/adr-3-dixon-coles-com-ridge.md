# ADR-3 — Dixon-Coles com peso temporal e penalidade ridge

**Status:** aceito
**Data:** 2026-10-19

## Contexto

Eu precisava de um modelo de PLACAR (nao so de resultado), porque a simulacao da
tabela precisa de saldo e gols pro (ADR-6). Dixon & Coles (1997) e o padrao da
literatura: gols de cada lado ~ Poisson com ataque/defesa por clube, mando,
correcao `rho` para placares baixos e peso exponencial no tempo.

Dois problemas praticos do modelo original:

1. **Identificabilidade:** somar `c` a todo ataque e subtrair de toda defesa da a
   mesma verossimilhanca. O artigo impoe "media dos ataques = 0".
2. **Clubes com poucos jogos** (recem-promovidos) ficam com parametros extremos.

## Decisao

Somar uma penalidade L2 `reg * (soma ataque^2 + soma defesa^2)` a
log-verossimilhanca negativa, com intercepto livre `mu0`. A penalidade resolve os
dois problemas de uma vez: fixa a escala e encolhe para a media quem tem pouca
evidencia.

Peso temporal `exp(-xi * dias)`. Hiperparametros escolhidos so na validacao
(2014-2016): `xi = 0,003` (meia-vida de ~230 dias) e `reg = 8`. A superficie de RPS
e suave e o otimo e interior — nao esta na borda da grade.

Ajuste por L-BFGS-B com **gradiente analitico**, conferido contra o numerico em
`test_gradiente_analitico_bate_com_o_numerico`. Um ajuste leva ~15 ms; o
walk-forward faz centenas deles.

## Consequencias

**A favor:** o teste com liga sintetica recupera os parametros verdadeiros
(correlacao > 0,9). `reg = 8` e um encolhimento forte — leitura: o Brasileirao e
equilibrado, e acreditar demais na diferenca entre clubes custa caro.

**Contra:** ridge e um vies deliberado: os melhores clubes ficam um pouco
subestimados e os piores, superestimados. A figura de calibracao mostra que esse
vies e pequeno na pratica.

## Alternativas consideradas

- **Restricao de soma zero (artigo original):** resolve identificabilidade, nao
  resolve clube com pouco dado.
- **Modelo bayesiano hierarquico (PyMC/Stan):** o equivalente "correto" do ridge,
  com a incerteza dos parametros de brinde. Mais lento e mais pesado de explicar;
  fica como evolucao.
- **Gradient boosting com features (forma, Elo...):** preve H/D/A, mas nao gera
  uma distribuicao de placar coerente para a simulacao.
