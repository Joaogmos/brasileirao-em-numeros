import itertools

import numpy as np
import pandas as pd
import pytest


def temporada_sintetica(clubes: list[str], temporada: int = 2030, semente: int = 0) -> pd.DataFrame:
    """Turno e returno completo entre `clubes`, no formato BRUTO da fonte."""
    rng = np.random.default_rng(semente)
    linhas = []
    inicio = pd.Timestamp(f"{temporada}-05-01")
    for i, (casa, fora) in enumerate(itertools.permutations(clubes, 2)):
        hg, ag = rng.poisson(1.4), rng.poisson(1.0)
        res = "H" if hg > ag else "A" if hg < ag else "D"
        dia = inicio + pd.Timedelta(days=i // 10)
        linhas.append({
            "Date": dia.strftime("%d/%m/%Y"), "Time": "20:00", "Season": temporada,
            "Home": casa, "Away": fora, "HG": hg, "AG": ag, "Res": res,
            "AvgCH": 2.10, "AvgCD": 3.30, "AvgCA": 3.60,
        })
    return pd.DataFrame(linhas)


@pytest.fixture
def clubes20() -> list[str]:
    return [f"Clube{i:02d}" for i in range(20)]


@pytest.fixture
def bruto(clubes20) -> pd.DataFrame:
    return pd.concat([temporada_sintetica(clubes20, 2030), temporada_sintetica(clubes20, 2031, 1)],
                     ignore_index=True)
