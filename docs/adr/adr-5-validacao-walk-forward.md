# ADR-5 — Validacao walk-forward com divisao temporal fixa

**Status:** aceito
**Data:** 2026-10-19

## Contexto

Em serie temporal, k-fold aleatorio vaza o futuro: o modelo treina com a rodada 30
para "prever" a rodada 10 e ja conhece a forca que o clube so mostraria depois. O
numero reportado sai otimista e nao vale na vida real.

## Decisao

- **Walk-forward semanal:** para cada semana, ajusta com todos os jogos ANTERIORES
  ao inicio dela e preve so a semana. O teste
  `test_walk_forward_nunca_treina_com_o_futuro_e_cobre_todos_os_jogos` usa um
  modelo-espiao que falha se enxergar um jogo do mesmo dia ou posterior.
- **Divisao fixa:** 2012-2013 aquecimento; 2014-2016 validacao (hiperparametros);
  2017-2026 teste, rodado com os hiperparametros congelados.

## Consequencias

A primeira versao validava so em 2014-2015. O Elo escolhido ali (`k = 5`) piorou no
teste — sobreajuste a 760 jogos. Com tres temporadas de validacao a escolha ficou
estavel. Fica registrado porque e exatamente o tipo de erro que esse desenho existe
para pegar.

`mando = 0` foi o melhor valor para o Elo: a vantagem de jogar em casa ja e
absorvida pelos limiares assimetricos do logit ordinal que converte rating em
probabilidade.

## Alternativas consideradas

- **k-fold aleatorio:** vazamento.
- **Reajustar a cada jogo:** mais fiel, ~10x mais lento, e os jogos de uma rodada
  acontecem praticamente juntos.
- **`TimeSeriesSplit` do scikit-learn:** mesma ideia, mas corta por numero de
  linhas, nao por semana — partiria uma rodada ao meio.
