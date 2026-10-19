# Dados

## Fonte

[football-data.co.uk](https://www.football-data.co.uk/brazil.php) — arquivo
`new/BRA.csv`, Campeonato Brasileiro Serie A de 2012 ate a rodada mais recente.
Dados gratuitos, mantidos por Joseph Buchdahl; usados aqui para fins educacionais,
com credito a fonte.

`raw/BRA.csv` e um **snapshot versionado**. `raw/MANIFEST.json` guarda o SHA-256 e a
data do download; `brasileirao.dados.carregar()` confere o hash antes de qualquer
analise, entao os numeros do README sao reproduziveis byte a byte. Para atualizar:

```bash
python -m brasileirao.dados --atualizar   # baixa, regrava o manifesto e revalida
```

## Dicionario (apos `limpar()`)

| Coluna | Tipo | Descricao |
|---|---|---|
| `temporada` | int | Ano da edicao (a de 2020 terminou em fev/2021, por causa da pandemia) |
| `data` | datetime | Data e hora do jogo (fuso de Londres, como na fonte) |
| `mandante`, `visitante` | str | Clube, com nome normalizado (`Flamengo RJ` -> `Flamengo`) |
| `gols_mandante`, `gols_visitante` | int | Placar final |
| `resultado` | `H`/`D`/`A` | Vitoria do mandante / empate / vitoria do visitante |
| `odd_mandante`, `odd_empate`, `odd_visitante` | float | Odd decimal **de fechamento**, media entre as casas |

## Particularidades encontradas na limpeza

- **Chapecoense x Atletico-MG, 11/12/2016** aparece sem placar: a ultima rodada de
  2016 nao foi disputada pela Chapecoense apos o acidente aereo de 29/11/2016.
  O jogo vai para `pendentes` e a temporada passa na validacao como 379 + 1.
- As colunas da Pinnacle (`PSC*`) faltam em 344 jogos (quase toda 2026); por isso
  o projeto usa a media de fechamento (`AvgC*`), presente em 100% das linhas.
- A margem da casa (soma de `1/odd`) fica entre 1,03 e 1,09 — `validar()` rejeita
  qualquer linha fora de [1,00; 1,25], que indicaria odd digitada errada.
