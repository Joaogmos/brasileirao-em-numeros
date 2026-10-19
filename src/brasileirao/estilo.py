"""Paleta e estilo unicos para todas as figuras (matplotlib).

Regra: cinza para o contexto, cor so para o que a figura quer mostrar. O Santos,
quando destacado, usa a tinta primaria (preto) -- as cores do clube.
"""

import matplotlib as mpl

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

AZUL = "#2a78d6"
LARANJA = "#eb6834"
VERDE_AGUA = "#1baf7a"
VERMELHO = "#d03b3b"
CINZA_CLARO = "#cfcdc6"

SANTOS = TINTA

# sequencial (uma cor, claro -> escuro) para heatmaps
AZUL_SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def aplicar() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": EIXO,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUDO,
        "ytick.color": MUDO,
        "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2,
        "lines.linewidth": 2,
        "legend.frameon": False,
    })
