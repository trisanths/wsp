"""The meta build must work as the first transformers model use in a fresh process."""

import subprocess
import sys
import textwrap


def test_meta_build_works_in_a_fresh_process():
    # Round 24's smoke died here: the model class was first imported inside
    # "with torch.device('meta')", where importing torch._dynamo fails.
    code = textwrap.dedent("""
        import transformers
        from wsp import parts
        config = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                          num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)
        names = parts.block_tensor_names(config, 0, 1)
        assert len(names) > 0
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    assert "ok" in out.stdout
