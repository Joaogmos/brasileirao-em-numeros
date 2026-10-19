# ADR-2 — O benchmark e a odd media de fechamento, sem margem

**Status:** aceito
**Data:** 2026-10-19

## Contexto

Bater um baseline ingenuo (frequencia historica) e facil e diz pouco. A pergunta
interessante e: o modelo sabe algo que o mercado nao sabe? O mercado de apostas
agrega informacao de muita gente com dinheiro em jogo (escalacao, lesao,
motivacao), e a odd de **fechamento** e a mais informada de todas.

## Decisao

- Usar a odd **media** de fechamento entre as casas (`AvgC*`). A da Pinnacle
  (`PSC*`), referencia mais afiada, falta em 344 jogos (quase toda 2026); a media
  esta em 100% das linhas e mantem o benchmark consistente no tempo.
- Remover a margem da casa (soma de 1/odd ~ 1,07) por normalizacao proporcional.
  O metodo de Shin tambem e calculado e reportado.

## Consequencias

Shin e marginalmente melhor (RPS -0,00014, IC95% [-0,00032; +0,00004]). O IC
cruza zero, entao a escolha nao muda nenhuma conclusao; fica o proporcional por
ser o mais simples de explicar.

O mercado vence o modelo em todas as 10 temporadas de teste. E o resultado
esperado (o modelo so ve placares) e e reportado como tal, nao escondido.

## Alternativas consideradas

- **So o baseline como referencia:** infla a percepcao de qualidade do modelo.
- **Pinnacle quando houver, media quando nao:** trocar de fonte no meio da serie
  mudaria o benchmark justamente no periodo mais recente.
