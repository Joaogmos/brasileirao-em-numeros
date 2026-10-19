# ADR-4 — Clube que sobe herda a forca de quem caiu

**Status:** aceito
**Data:** 2026-10-19

## Contexto

O Mirassol (2025) nunca tinha jogado a Serie A. Sem dados, a penalidade ridge
deixaria os parametros dele em 0 — "time medio". Mas quem sobe da Serie B costuma
ser bem mais fraco que a media da Serie A.

## Decisao

- **Dixon-Coles:** clube sem historico recebe a media de ataque/defesa dos 4
  clubes mais fracos da ultima temporada do historico — justamente os que ele
  substitui.
- **Elo:** na virada de temporada todos regridem parte do caminho para a media;
  quem entra herda a media dos ratings de quem saiu; quem volta depois de anos fica
  com o menor entre o rating antigo e essa heranca.

## Consequencias

Heuristica simples, listada como limitacao no README. O teste
`test_clube_desconhecido_recebe_parametros_de_clube_fraco` garante o comportamento.

## Alternativas consideradas

- **Usar dados da Serie B:** a fonte gratuita nao tem, e exigiria modelar a
  diferenca de nivel entre divisoes.
- **Parametro "promovido" estimado:** mais correto, mas com ~4 promovidos por ano a
  estimativa seria ruidosa.
