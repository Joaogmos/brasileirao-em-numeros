"""Caminhos e constantes do projeto. Um lugar so para nao espalhar numero magico."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
RELATORIOS = RAIZ / "reports"
FIGURAS = RELATORIOS / "figures"

# Fonte: football-data.co.uk ("new leagues"), atualizada semanalmente.
URL_FONTE = "https://www.football-data.co.uk/new/BRA.csv"
ARQUIVO_BRUTO = DADOS_BRUTOS / "BRA.csv"
MANIFESTO = DADOS_BRUTOS / "MANIFEST.json"

# Formato da Serie A desde 2006: 20 clubes, pontos corridos, turno e returno.
CLUBES_POR_TEMPORADA = 20
JOGOS_POR_TEMPORADA = CLUBES_POR_TEMPORADA * (CLUBES_POR_TEMPORADA - 1)  # 380

# Resultado sempre na otica do mandante: H = vitoria mandante, D = empate, A = vitoria visitante.
RESULTADOS = ("H", "D", "A")

SEMENTE = 2026
