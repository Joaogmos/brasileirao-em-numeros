"""Painel interativo do projeto.

    streamlit run app/streamlit_app.py

Le os resultados que o pipeline grava em reports/ e ajusta o Dixon-Coles na
inicializacao (leva ~0,1 s) para o simulador de confrontos.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from brasileirao import config
from brasileirao.analises.projecao import projetar
from brasileirao.analises.santos import CLUBE
from brasileirao.dados import carregar
from brasileirao.simulacao import zonas

st.set_page_config(page_title="Brasileirao em numeros", page_icon="⚽", layout="wide")


def ler(nome: str) -> dict:
    return json.loads((config.RELATORIOS / nome).read_text(encoding="utf-8"))


@st.cache_resource
def base():
    partidas = carregar().partidas
    sim = projetar(partidas, n=2000)  # so para ter o modelo ajustado e os jogos restantes
    return partidas, sim


partidas, sim0 = base()
modelo = sim0["modelo"]
temporada = sim0["temporada"]
clubes = sorted(set(sim0["jogados"]["mandante"]))

st.title("Brasileirao em numeros")
st.caption(
    f"Modelo Dixon-Coles avaliado contra o mercado, com foco no Santos FC. "
    f"Dados de {partidas['temporada'].min()} a {sim0['ultima_data']:%d/%m/%Y} (football-data.co.uk)."
)

aba_proj, aba_jogo, aba_modelo, aba_2023, aba_mando = st.tabs(
    [f"Projecao {temporada}", "Simulador de confronto", "Modelo x mercado", "Santos 2023", "Mando de campo"]
)

# ------------------------------------------------------------------ projecao
with aba_proj:
    pr = ler("projecao.json")
    s = pr["santos"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posicao atual do Santos", f"{s['posicao']}o", f"{s['pontos']} pts em {s['jogos']} jogos",
              delta_color="off")
    c2.metric("Pontos esperados ao final", f"{s['pontos_esperados']:.0f}",
              f"IC90%: {s['pontos_p05']:.0f} a {s['pontos_p95']:.0f}", delta_color="off")
    c3.metric("Chance de top 6", f"{s['top6']:.0%}")
    c4.metric("Chance de rebaixamento", f"{s['rebaixamento']:.1%}")

    st.image(str(config.FIGURAS / "projecao_heatmap.png"), width="stretch")

    with st.expander("Rodar uma nova simulacao"):
        n = st.select_slider("Numero de temporadas simuladas", [1000, 5000, 10000, 20000], value=5000)
        if st.button("Simular"):
            with st.spinner("simulando..."):
                nova = projetar(partidas, n=n)
            z = zonas(nova["posicoes"])
            tabela = nova["resumo"].join(z).sort_values("posicao_media")
            st.dataframe(
                tabela.style.format({"pontos_atuais": "{:.0f}", "pontos_esperados": "{:.1f}",
                                     "pontos_p05": "{:.0f}", "pontos_p95": "{:.0f}",
                                     "posicao_media": "{:.1f}", "titulo": "{:.1%}", "top6": "{:.1%}",
                                     "rebaixamento": "{:.1%}"}),
                width="stretch",
            )

    v = pr["validacao_no_passado"]["rebaixado"]["entre_10_e_90pct"]
    st.info(
        f"**Da para confiar?** A mesma projecao, refeita com 300 jogos disputados em 2017-2025, deu em "
        f"media {v['prob_media']:.0%} de risco aos clubes na zona de duvida -- e {v['taxa_real']:.0%} deles "
        "cairam de fato."
    )

# ------------------------------------------------------------------ confronto
with aba_jogo:
    st.subheader("Quem vence? Probabilidades do Dixon-Coles para qualquer confronto")
    col_a, col_b = st.columns(2)
    mandante = col_a.selectbox("Mandante", clubes, index=clubes.index(CLUBE))
    outros = [c for c in clubes if c != mandante]
    padrao = outros.index("Palmeiras") if "Palmeiras" in outros else 0
    visitante = col_b.selectbox("Visitante", outros, index=padrao)

    jogo = pd.DataFrame({"mandante": [mandante], "visitante": [visitante]})
    lam, mu = modelo.taxas(jogo)
    m = modelo.matriz_placares(lam, mu)[0]
    ph, pd_, pa = np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Vitoria {mandante}", f"{ph:.0%}")
    k2.metric("Empate", f"{pd_:.0%}")
    k3.metric(f"Vitoria {visitante}", f"{pa:.0%}")
    k4.metric("Gols esperados", f"{lam[0]:.2f} x {mu[0]:.2f}")

    top = sorted(((m[i, j], f"{i} x {j}") for i in range(6) for j in range(6)), reverse=True)[:6]
    st.write("**Placares mais provaveis:** " + " · ".join(f"{p} ({v:.1%})" for v, p in top))

    grade = pd.DataFrame(m[:6, :6] * 100, index=[f"{i}" for i in range(6)], columns=[f"{j}" for j in range(6)])
    grade.index.name = f"gols {mandante}"
    grade.columns.name = f"gols {visitante}"
    st.dataframe(grade.style.format("{:.1f}%").background_gradient(cmap="Blues", axis=None),
                 width="stretch")
    st.caption("Forcas estimadas com todos os jogos ate a ultima rodada, jogos recentes pesando mais.")

# ------------------------------------------------------------------ modelo x mercado
with aba_modelo:
    met = ler("metricas.json")
    st.subheader(f"Validacao walk-forward: {met['n_jogos']} jogos, {met['temporadas'][0]}-{met['temporadas'][1]}")
    st.dataframe(pd.DataFrame(met["geral"]).T.style.format("{:.4f}"), width="stretch")
    comp = pd.DataFrame(met["comparacoes_rps"]).T[["diferenca", "ic_inf", "ic_sup"]]
    st.write("Diferenca de RPS (negativo = o primeiro e melhor), IC95% por bootstrap pareado:")
    st.dataframe(comp.style.format("{:+.5f}"), width="stretch")
    a, b = st.columns(2)
    a.image(str(config.FIGURAS / "rps_modelos.png"), width="stretch")
    b.image(str(config.FIGURAS / "calibracao.png"), width="stretch")
    st.image(str(config.FIGURAS / "rps_por_temporada.png"), width="stretch")

# ------------------------------------------------------------------ santos 2023
with aba_2023:
    s23 = ler("santos.json")
    st.subheader("O rebaixamento de 2023 foi azar?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pontos reais", s23["pontos_reais"])
    c2.metric("Pontos esperados pelas odds", f"{s23['xpts_mercado']:.1f}")
    c3.metric("Simulacoes com 43 pts ou menos", f"{s23['p_pontos_ate_o_real']:.0%}")
    st.image(str(config.FIGURAS / "santos_2023_simulacao.png"), width="stretch")
    pre = s23["pre_temporada_dixon_coles"]
    st.markdown(
        f"Antes da 1a rodada o modelo via o Santos como **{pre['ranking_forca_pre_temporada']}o** em forca "
        f"({pre['p_rebaixamento']:.0%} de risco de queda). Ao longo do ano o mercado foi rebaixando a "
        f"expectativa jogo a jogo -- o Santos foi favorito em so **{s23['jogos_como_favorito']}** dos 38 jogos "
        "-- e o resultado final ficou dentro do esperado. **Nao foi azar: foi queda de nivel.**"
    )
    st.image(str(config.FIGURAS / "santos_pontos_vs_xpts.png"), width="stretch")

# ------------------------------------------------------------------ mando
with aba_mando:
    mj = ler("mando.json")["portoes_fechados"]
    st.subheader("Quanto vale jogar em casa? O que 2020 (sem publico) mostrou")
    c1, c2, c3 = st.columns(3)
    c1.metric("Saldo do mandante com publico (2012-19)", f"{mj['saldo_com_publico']:+.2f}")
    c2.metric("Sem publico (2020)", f"{mj['saldo_sem_publico']:+.2f}", f"{mj['queda_relativa']:.0%}")
    c3.metric("p-valor (permutacao)", f"{mj['p_valor_permutacao']:.3f}")
    st.image(str(config.FIGURAS / "mando_por_temporada.png"), width="stretch")
    st.caption("Ressalva: 2017, com torcida, teve queda do mesmo tamanho. Uma temporada so nao prova causalidade.")
