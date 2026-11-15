import subprocess
import sys


def test_parts_imports_without_transformers():
    """wsp.parts must import in an environment without Transformers (the studio server imports it)."""
    code = ("import sys; sys.modules['transformers'] = None\n"
            "import wsp.parts\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
