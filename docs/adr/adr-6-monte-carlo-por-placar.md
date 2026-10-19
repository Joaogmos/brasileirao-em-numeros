# ADR-6 — A simulacao sorteia placares, nao so resultados

**Status:** aceito
**Data:** 2026-10-19

## Contexto

Projetar "chance de cair" exige rejogar o resto da temporada muitas vezes. Sortear
so vitoria/empate/derrota e mais simples, mas a tabela desempata por vitorias,
**saldo** e **gols pro** — sem placar, esses criterios viram sorteio.

## Decisao

Para cada jogo restante, sortear uma celula da matriz P(placar = i x j) do
Dixon-Coles (11 x 11, ate 10 gols) por inversao da CDF, vetorizado nas N
simulacoes. Ordenar com `np.lexsort` por pontos, vitorias, saldo e gols pro; sorteio
so para o que sobrar.

O calendario restante nao vem na fonte, mas e deduzivel: turno e returno = todo par
ordenado (mandante, visitante) exatamente uma vez.

## Consequencias

20 mil temporadas em ~2 s. Validacao retroativa (2017-2025, projetando com 300
jogos disputados): clubes com 10-90% de risco de queda tinham 46% em media e 45%
cairam; os 16 com >= 90% cairam todos.

**Limitacao:** os parametros do modelo sao os mesmos em todas as simulacoes (sem
incerteza de estimacao), o que deixa as pontas um pouco confiantes demais. A
correcao seria reamostrar os parametros (bootstrap ou posterior bayesiana) a cada
simulacao.

## Alternativas consideradas

- **Sortear H/D/A com as odds:** usado na analise do Santos 2023, onde o ponto e
  justamente comparar com o mercado; perde os desempates.
- **Calcular a distribuicao exata:** combinatoria inviavel com ~80 jogos restantes.
