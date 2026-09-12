from streamlit.testing.v1 import AppTest

from core.config import ROOT


def test_ui_query_and_mode_switch():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert next(s for s in app.selectbox if s.label == "Backend").value == "huggingface"
    assert next(s for s in app.selectbox if s.label == "Phương pháp").value == "greedy"
    next(s for s in app.selectbox if s.label == "Backend").select("mock").run()
    next(b for b in app.button if b.label == "Dùng câu hỏi mẫu").click().run()
    next(b for b in app.button if b.label == "Generate SQL").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["last_result"]["selected"]
    method = next(s for s in app.selectbox if s.label == "Phương pháp")
    method.select("fixed_sc").run()
    next(b for b in app.button if b.label == "Generate SQL").click().run()
    assert len(app.session_state["last_result"]["candidates"]) == 8


def test_ui_empty_question_is_explained():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    next(b for b in app.button if b.label == "Generate SQL").click().run()
    assert not app.exception
    assert app.error
