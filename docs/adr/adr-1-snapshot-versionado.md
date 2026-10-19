# ADR-1 — Os dados brutos sao um snapshot versionado com SHA-256

**Status:** aceito
**Data:** 2026-10-19

## Contexto

A fonte (football-data.co.uk) e atualizada toda semana e nao tem versao. Se o
pipeline baixasse o CSV a cada execucao, os numeros do README mudariam sozinhos,
o CI dependeria de rede e ninguem conseguiria reproduzir a analise de uma data
especifica.

## Decisao

`data/raw/BRA.csv` e versionado no git (645 KB) junto com `MANIFEST.json`
(SHA-256, tamanho e data do download). `dados.carregar()` recalcula o hash e falha
se nao bater. Atualizar e um ato explicito: `python -m brasileirao.dados --atualizar`.

`.gitattributes` marca `data/raw/**` como `-text`: sem isso o git converteria o fim
de linha no checkout e o hash mudaria conforme o sistema operacional.

## Consequencias

**A favor:** reproducao byte a byte; CI sem internet; o diff do git mostra
exatamente quais jogos entraram entre duas versoes.

**Contra:** dado dentro do repositorio. Aceitavel com 645 KB; com GBs eu usaria
DVC ou um bucket com o hash no manifesto — a mesma ideia com outra infraestrutura.

## Alternativas consideradas

- **Baixar sempre:** resultados nao reprodutiveis e CI fragil.
- **DVC / Git LFS:** resolvem o mesmo problema com mais infraestrutura do que o
  tamanho do dado justifica.
