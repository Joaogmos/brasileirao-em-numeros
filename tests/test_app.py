import pytest

from brasileirao import config

pytest.importorskip("streamlit")


@pytest.mark.skipif(not (config.RELATORIOS / "projecao.json").exists(), reason="rode o pipeline antes")
def test_painel_abre_sem_erro_e_simulador_responde():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.RAIZ / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    at.selectbox[0].select("Santos").run()
    at.selectbox[1].select("Flamengo").run()
    probs = [m.value for m in at.metric if m.label.startswith("Vitoria") or m.label == "Empate"]
    assert len(probs) == 3 and all(v.endswith("%") for v in probs)
