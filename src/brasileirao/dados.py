"""Ingestao, limpeza e validacao dos dados de partidas.

Fluxo:  BRA.csv (bruto, versionado)  ->  limpar()  ->  validar()  ->  DataFrame de partidas

O CSV bruto fica versionado em data/raw/ junto com um MANIFEST.json (SHA-256 +
data do download). Assim qualquer pessoa reproduz exatamente os numeros do
README, mesmo que a fonte seja atualizada na semana seguinte (ADR-1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from brasileirao import config

# Nomes da fonte -> nome usado no projeto. Remove sufixos de estado quando nao ha ambiguidade.
NOMES_CLUBES = {
    "Flamengo RJ": "Flamengo",
    "Botafogo RJ": "Botafogo",
    "Chapecoense-SC": "Chapecoense",
    "Sport Recife": "Sport",
    "America MG": "America-MG",
    "Atletico GO": "Atletico-GO",
}

COLUNAS = {
    "Season": "temporada",
    "Home": "mandante",
    "Away": "visitante",
    "HG": "gols_mandante",
    "AG": "gols_visitante",
    "Res": "resultado",
    # Odds de fechamento MEDIAS entre as casas. A Pinnacle (PSC*) seria a referencia
    # mais "afiada", mas falta em 344 jogos (quase toda 2026) -- ver ADR-2.
    "AvgCH": "odd_mandante",
    "AvgCD": "odd_empate",
    "AvgCA": "odd_visitante",
}


class ErroDeValidacao(ValueError):
    """Os dados violam alguma regra do dominio. Carrega a lista completa de problemas."""

    def __init__(self, problemas: list[str]):
        self.problemas = problemas
        super().__init__("\n".join(problemas))


@dataclass(frozen=True)
class Dados:
    partidas: pd.DataFrame  # jogos com placar
    pendentes: pd.DataFrame  # jogos listados na fonte sem placar (nao realizados)


# --------------------------------------------------------------------------- download


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 16), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar(url: str = config.URL_FONTE, destino: Path = config.ARQUIVO_BRUTO) -> dict:
    """Baixa o CSV da fonte e grava o manifesto. So roda quando pedido (--atualizar)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (URL fixa)
        destino.write_bytes(resp.read())
    manifesto = {
        "fonte": url,
        "baixado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": sha256(destino),
        "bytes": destino.stat().st_size,
    }
    config.MANIFESTO.write_text(json.dumps(manifesto, indent=2) + "\n", encoding="utf-8")
    return manifesto


def conferir_manifesto(caminho: Path = config.ARQUIVO_BRUTO) -> None:
    """Falha alto se o arquivo bruto nao for o mesmo que gerou os resultados versionados."""
    esperado = json.loads(config.MANIFESTO.read_text(encoding="utf-8"))["sha256"]
    obtido = sha256(caminho)
    if obtido != esperado:
        raise ErroDeValidacao(
            [f"SHA-256 de {caminho.name} nao bate com o MANIFEST.json ({obtido[:12]} != {esperado[:12]})"]
        )


# --------------------------------------------------------------------------- limpeza


def ler_bruto(caminho: Path = config.ARQUIVO_BRUTO) -> pd.DataFrame:
    return pd.read_csv(caminho, encoding="utf-8-sig")  # a fonte vem com BOM


def limpar(bruto: pd.DataFrame) -> Dados:
    df = bruto[["Date", "Time", *COLUNAS]].rename(columns=COLUNAS)

    # A fonte grava data/hora no fuso de Londres; para ordenar no tempo isso basta.
    df["data"] = pd.to_datetime(df.pop("Date") + " " + df.pop("Time"), format="%d/%m/%Y %H:%M")
    df["mandante"] = df["mandante"].replace(NOMES_CLUBES)
    df["visitante"] = df["visitante"].replace(NOMES_CLUBES)

    sem_placar = df["gols_mandante"].isna() | df["gols_visitante"].isna()
    pendentes = df.loc[sem_placar, ["temporada", "data", "mandante", "visitante"]]

    partidas = df.loc[~sem_placar].copy()
    partidas["gols_mandante"] = partidas["gols_mandante"].astype(int)
    partidas["gols_visitante"] = partidas["gols_visitante"].astype(int)
    partidas = partidas.sort_values(["data", "mandante"], kind="stable").reset_index(drop=True)

    colunas = ["temporada", "data", "mandante", "visitante", "gols_mandante", "gols_visitante",
               "resultado", "odd_mandante", "odd_empate", "odd_visitante"]
    return Dados(partidas=partidas[colunas], pendentes=pendentes.reset_index(drop=True))


# --------------------------------------------------------------------------- validacao


def validar(dados: Dados, temporada_atual: int | None = None) -> None:
    """Regras de dominio da Serie A. Junta TODOS os problemas antes de falhar.

    Temporadas encerradas: 20 clubes, 380 jogos (ou 379 + 1 pendente), 19 jogos em
    casa e 19 fora por clube. A temporada atual (incompleta) so precisa nao estourar.
    """
    p = dados.partidas
    problemas: list[str] = []
    atual = temporada_atual if temporada_atual is not None else int(p["temporada"].max())

    duplicados = p.duplicated(["temporada", "mandante", "visitante"])
    if duplicados.any():
        problemas.append(f"{duplicados.sum()} confronto(s) mandante x visitante repetido(s) na mesma temporada")

    if (p["mandante"] == p["visitante"]).any():
        problemas.append("clube jogando contra si mesmo")

    if (p[["gols_mandante", "gols_visitante"]] < 0).any().any():
        problemas.append("placar negativo")

    esperado = np.select(
        [p["gols_mandante"] > p["gols_visitante"], p["gols_mandante"] < p["gols_visitante"]],
        ["H", "A"],
        "D",
    )
    inconsistentes = int((esperado != p["resultado"]).sum())
    if inconsistentes:
        problemas.append(f"{inconsistentes} jogo(s) com 'resultado' diferente do placar")

    odds = p[["odd_mandante", "odd_empate", "odd_visitante"]]
    if odds.isna().any().any() or (odds <= 1).any().any():
        problemas.append("odd ausente ou <= 1")
    else:
        margem = (1 / odds).sum(axis=1)
        if not margem.between(1.0, 1.25).all():
            problemas.append("margem da casa (soma de 1/odd) fora de [1.00, 1.25]")

    for temporada, jogos in p.groupby("temporada"):
        clubes = set(jogos["mandante"]) | set(jogos["visitante"])
        if len(clubes) != config.CLUBES_POR_TEMPORADA:
            problemas.append(f"{temporada}: {len(clubes)} clubes (esperado 20)")
        casa = jogos["mandante"].value_counts()
        fora = jogos["visitante"].value_counts()
        if casa.max() > 19 or fora.max() > 19:
            problemas.append(f"{temporada}: clube com mais de 19 jogos como mandante ou visitante")
        if temporada != atual:
            n_pend = int((dados.pendentes["temporada"] == temporada).sum())
            if len(jogos) + n_pend != config.JOGOS_POR_TEMPORADA:
                problemas.append(f"{temporada}: {len(jogos)} jogos + {n_pend} pendente(s) != 380")

    if problemas:
        raise ErroDeValidacao(problemas)


def carregar(caminho: Path = config.ARQUIVO_BRUTO, conferir: bool = True) -> Dados:
    """Ponto de entrada usado pelo resto do projeto: le, limpa e valida."""
    if conferir and caminho == config.ARQUIVO_BRUTO:
        conferir_manifesto(caminho)
    dados = limpar(ler_bruto(caminho))
    validar(dados)
    return dados


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingestao dos dados da Serie A")
    parser.add_argument("--atualizar", action="store_true", help="baixa a versao mais nova da fonte")
    args = parser.parse_args(argv)

    if args.atualizar:
        m = baixar()
        print(f"baixado: {m['bytes']} bytes, sha256 {m['sha256'][:12]}")

    dados = carregar()
    p = dados.partidas
    print(f"{len(p)} partidas validas | {p['temporada'].min()}-{p['temporada'].max()} "
          f"| {len(dados.pendentes)} pendente(s)")
    config.DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    p.to_csv(config.DADOS_PROCESSADOS / "partidas.csv", index=False)


if __name__ == "__main__":
    main()
