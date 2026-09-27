import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_defines_main_and_tabs():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for required in ("main", "qa_tab", "summary_tab", "sources_tab", "sidebar"):
        assert required in names
    assert "Question answering" in source
    assert "Summary" in source
    assert "file_uploader" in source
    assert "Process documents" in source


def test_app_does_not_hardcode_api_key():
    root = ROOT
    for path in root.rglob("*.py"):
        if "tests" in path.parts or ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "AIza" not in text
        assert "GEMINI_API_KEY=" not in text.replace("os.getenv(\"GEMINI_API_KEY\"", "")
