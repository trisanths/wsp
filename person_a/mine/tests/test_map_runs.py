# CPU test of how maps.py and maps_many.py run a --modules job: the skip rule, the entry write
# guard, the split of the module passes over engine calls, the saved state between calls, the
# old-engine check, and the failure memory of maps_many.py. No Modal call.
#   .venv/bin/python mine/tests/test_map_runs.py
import contextlib
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import maps  # noqa: E402
import maps_many  # noqa: E402
import test_module_maps as synthetic  # noqa: E402

N = synthetic.N
STAMP = {"model": "org/toy", "revision": "abc", "layers": N, "skills": {"law": "h1"}}


def module_call(layers):
    # what the engine returns for skill_maps(payload, modules=True, layers=layers, plain=False)
    res = synthetic.engine_result(True)
    res["layer_lp"] = None
    res["layers"] = sorted(layers)
    for module in ["attn_lp", "mlp_lp"]:
        for key in res[module]:
            res[module][key] = [v if l in layers else None for l, v in enumerate(res[module][key])]
    return res


def write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def test_entry_done():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "org__toy.json")
        # measured with --modules, and no skill got a module map: done, not paid for again
        write_json(path, {"module_maps": {}, "layer_maps": {}})
        assert maps.entry_done(path, True) is True and maps.entry_done(path, False) is True
        write_json(path, {"layer_maps": {}})
        assert maps.entry_done(path, True) is False and maps.entry_done(path, False) is True
        write_json(path, ["not", "an", "entry"])
        assert maps.entry_done(path, True) is False and maps.read_entry(path) is None
        assert maps.read_entry(os.path.join(tmp, "missing.json")) is None


def test_write_entry():
    plain = {"model": "org/toy", "layer_maps": {"law": {}}, "revision": "abc"}
    full = dict(plain, module_maps={"law": {}})
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "entries", "org__toy.json")
        assert maps.write_entry(path, plain) is True and read_json(path) == plain
        assert maps.write_entry(path, full) is True and read_json(path) == full
        # a plain run that ends after the --modules run of the same model does not undo it
        assert maps.write_entry(path, dict(plain, revision="later")) is False
        assert read_json(path) == full
        # an empty module_maps is still the result of a --modules run
        write_json(path, dict(plain, module_maps={}))
        assert maps.write_entry(path, plain) is False and "module_maps" in read_json(path)
        # --force replaces it
        assert maps.write_entry(path, dict(plain, revision="later"), True) is True
        assert read_json(path) == dict(plain, revision="later")
        # the tmp file of another process that writes the same entry is not touched, and no tmp stays
        other_tmp = path + ".tmp"
        with open(other_tmp, "w") as f:
            f.write("half of another run's entry")
        assert maps.write_entry(path, full) is True
        with open(other_tmp) as f:
            assert f.read() == "half of another run's entry"
        assert sorted(os.listdir(os.path.dirname(path))) == ["org__toy.json", "org__toy.json.tmp"]
        # a file that is not JSON is replaced
        with open(path, "w") as f:
            f.write("{ not json")
        assert maps.write_entry(path, plain) is True and read_json(path) == plain


def test_plan_module_calls():
    budget = maps.CALL_TIMEOUT * maps.CALL_BUDGET
    assert budget == 1800
    # the three plain entries on main: engine_seconds over (1 + layers) passes
    assert maps.seconds_per_pass({"engine_seconds": 909.3, "layers": 16}) == 909.3 / 17
    assert maps.seconds_per_pass({"layers": 16}) is None and maps.seconds_per_pass({"engine_seconds": 5.0}) is None
    # 53.49 s per pass: 1800 / 53.49 = 33.65 passes, (33.65 - 1) / 2 = 16.3 layers: one call for 16 layers
    assert maps.plan_module_calls(16, 909.3 / 17) == [list(range(16))]
    # 100 s per pass: (18 - 1) / 2 = 8.5, so 8 layers per call; 28 layers make 4 calls
    chunks = maps.plan_module_calls(28, 100.0)
    assert [len(c) for c in chunks] == [8, 8, 8, 4]
    assert [l for c in chunks for l in c] == list(range(28))
    # every planned call stays inside the budget by the estimate, or is a single layer
    for per_pass in [0.5, 7.0, 53.5, 100.0, 250.0, 599.0, 600.0, 601.0, 1200.0]:
        chunks = maps.plan_module_calls(28, per_pass)
        assert [l for c in chunks for l in c] == list(range(28)), per_pass
        for c in chunks:
            assert (1 + 2 * len(c)) * per_pass <= budget or len(c) == 1, (per_pass, len(c))
    assert len(maps.plan_module_calls(28, 0.5)) == 1
    # 600 s per pass: 3 passes are exactly the budget, 1 layer per call; 5 passes would not be
    assert maps.plan_module_calls(3, 600.0) == [[0], [1], [2]]
    # 1200 s per pass: the 3 passes of one layer just fit the timeout; above that the run is refused
    assert maps.plan_module_calls(2, 1200.0) == [[0], [1]]
    assert maps.plan_module_calls(2, 1200.1) is None
    assert maps.plan_module_calls(2, 1200.1, 1) is None
    # no time known: the smallest calls
    assert maps.plan_module_calls(3, None) == [[0], [1], [2]]
    # the user's layer count per call
    assert maps.plan_module_calls(7, 100.0, 3) == [[0, 1, 2], [3, 4, 5], [6]]
    assert maps.plan_module_calls(4, None, 9) == [[0, 1, 2, 3]]


def test_measure_modules_saves_each_call():
    whole = maps.module_increases(synthetic.engine_result(True))
    chunks = maps.plan_module_calls(N, None, 3)
    assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7]]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "logs", "org__toy.modules-partial.json")
        asked = []

        def breaks_on_second(layers):
            asked.append(list(layers))
            if len(asked) == 2:
                raise RuntimeError("the call reached its timeout")
            return module_call(layers)
        try:
            maps.measure_modules(breaks_on_second, chunks, path, STAMP)
        except RuntimeError as e:
            assert "timeout" in str(e) and not maps.is_old_engine_error(e)
        else:
            raise AssertionError("the failure of a call was swallowed")
        # the first call is on disk
        saved = read_json(path)
        assert saved["stamp"] == STAMP and saved["calls"] == 1 and saved["increases"]["layers"] == [0, 1, 2]
        assert saved["engine_seconds"] == 12.5
        assert os.listdir(os.path.dirname(path)) == ["org__toy.modules-partial.json"]

        # the next run asks only for what is missing
        asked = []

        def works(layers):
            asked.append(list(layers))
            return module_call(layers)
        state = maps.measure_modules(works, chunks, path, STAMP)
        assert asked == [[3, 4, 5], [6, 7]] and state["calls"] == 3 and state["engine_seconds"] == 37.5
        assert state["increases"]["layers"] == list(range(N))
        assert maps.module_fields(state["increases"], synthetic.SOURCES) == maps.module_fields(whole, synthetic.SOURCES)
        assert read_json(path)["increases"]["layers"] == list(range(N))

        # everything is saved: a run that died before it wrote the entry makes no call
        asked = []
        state = maps.measure_modules(works, chunks, path, STAMP)
        assert asked == [] and state["increases"]["layers"] == list(range(N))
        # other chunk sizes use the saved layers too
        write_json(path, saved)
        asked = []
        maps.measure_modules(works, [[0, 1, 2, 3, 4], [5, 6, 7]], path, STAMP)
        assert asked == [[3, 4], [5, 6, 7]]

        # a saved state of another revision or of other skill sets is not used
        for other in [dict(STAMP, revision="def"), dict(STAMP, skills={"law": "h2"}), dict(STAMP, layers=N + 1)]:
            write_json(path, saved)
            asked = []
            maps.measure_modules(works, chunks, path, other)
            assert asked == chunks, other

        # an engine that does not answer for the layers it was asked for
        os.remove(path)
        try:
            maps.measure_modules(lambda layers: module_call([0, 1]), chunks, path, STAMP)
        except RuntimeError as e:
            assert "asked for layers [0, 1, 2]" in str(e)
        else:
            raise AssertionError("a result for other layers was merged")
        assert not os.path.exists(path)


def test_old_engine():
    class MainEngine:
        # the signature that the engine on main has now
        def skill_maps(self, payload, routing_items=64):
            return {"base": {}, "layer_lp": {}}

    class LastCommitEngine:
        # the signature of the first commit of this branch
        def skill_maps(self, payload, routing_items=64, modules=False):
            return {"base": {}, "layer_lp": {}}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "partial.json")
        for engine in [MainEngine(), LastCommitEngine()]:
            def call(layers):
                return engine.skill_maps({}, modules=True, layers=layers, plain=False)
            try:
                maps.measure_modules(call, [[0], [1]], path, STAMP)
            except TypeError as e:
                assert maps.is_old_engine_error(e), str(e)
            else:
                raise AssertionError("the old engine took the call")
            assert not os.path.exists(path)
        # an engine that takes the call and returns a result without the module fields
        for res in [synthetic.engine_result(False), synthetic.engine_result(True)]:
            try:
                maps.measure_modules(lambda layers: res, [[0], [1]], path, STAMP)
            except maps.OldEngine as e:
                assert maps.is_old_engine_error(e)
            else:
                raise AssertionError("a result without module data for a layer list was used")
    assert not maps.is_old_engine_error(RuntimeError("CUDA out of memory"))
    assert not maps.is_old_engine_error(TimeoutError("the call reached its timeout"))
    assert maps.OLD_ENGINE_EXIT not in [0, 1, 2]  # 1: any failure, 2: argparse


def test_plain_entry_fits_and_added_fields():
    sources = synthetic.SOURCES
    plain = maps.build_entry("org/toy", synthetic.engine_result(False), sources, synthetic.ROW)
    plain = json.loads(json.dumps(plain))
    assert maps.plain_entry_fits(plain, "org/toy", "abc", sources)
    assert not maps.plain_entry_fits(plain, "org/toy", "def", sources)
    assert not maps.plain_entry_fits(plain, "org/other", "abc", sources)
    assert not maps.plain_entry_fits(plain, "org/toy", "abc", dict(sources, law="another law set"))
    assert not maps.plain_entry_fits(None, "org/toy", "abc", sources)
    assert not maps.plain_entry_fits({"model": "org/toy", "revision": "abc", "selection_data": sources}, "org/toy",
                                     "abc", sources)
    # an entry as main wrote it before this branch: it fits, and it gets the fields that need no GPU
    old = dict(plain)
    del old["selection_sources"]
    del old["reported_benchmark_skills"]
    assert maps.plain_entry_fits(old, "org/toy", "abc", sources)
    state = {"increases": maps.module_increases(synthetic.engine_result(True)), "engine_seconds": 30.5, "calls": 2}
    details = {"law": {"dataset": "cais/mmlu all test", "description": "d", "pool": None, "reported_benchmark": True}}
    out = maps.add_module_fields(old, state, sources, details)
    for key in old:
        assert out[key] == old[key], key  # what the plain run measured stays as it is
    assert sorted(set(out.keys()) - set(old.keys())) == [
        "module_calls", "module_engine_seconds", "module_maps", "module_measured_at", "module_method",
        "module_parts", "reported_benchmark_skills", "selection_sources"]
    assert out["module_engine_seconds"] == 30.5 and out["module_calls"] == 2
    assert out["selection_sources"] == details and out["reported_benchmark_skills"] == ["law"]
    assert out["module_maps"]["law"]["attn"] == [0.2, 0.0, 0.4, None, -0.2, 0.0, 0.3, 0.0]
    assert "module_maps" not in old
    # an entry that has selection_sources keeps its own
    mine = dict(plain, selection_sources={"law": {"reported_benchmark": False}}, reported_benchmark_skills=[])
    out = maps.add_module_fields(mine, state, sources, details)
    assert out["selection_sources"] == {"law": {"reported_benchmark": False}} and out["reported_benchmark_skills"] == []

    # An entry that stores the hashes of its skill sets fits only a run with the same sets: the pool
    # file built again with other items keeps the description strings.
    hashes = {"law": "maps-h1", maps.REFERENCE: "maps-builtin-general_text-aaaa"}
    stored = dict(plain, skill_hashes=dict(hashes))
    assert maps.plain_entry_fits(stored, "org/toy", "abc", sources, hashes)
    assert not maps.plain_entry_fits(stored, "org/toy", "abc", sources, dict(hashes, law="maps-h2"))
    assert maps.plain_entry_fits(stored, "org/toy", "abc", sources)          # a caller with no hashes
    assert maps.plain_entry_fits(plain, "org/toy", "abc", sources, hashes)   # an entry from before the field
    assert maps.skill_hashes({"skills": {"law": {"hash": "maps-h1", "items": {}}}}) == {"law": "maps-h1"}
    # the built-in sets of the real payload are named by their items, the pool sets by the pool hash
    payload, real_sources = maps.load_skills()
    for name in ["math", "code", maps.REFERENCE]:
        spec = payload["skills"][name]
        assert spec["hash"].startswith("maps-builtin-" + name + "-") and len(spec["hash"].rsplit("-", 1)[1]) == 12
        changed = json.loads(json.dumps(spec["items"]))
        changed["search"][0]["continuation"] += " x"
        digest = maps.hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).hexdigest()[:12]
        assert not spec["hash"].endswith(digest)
    assert sorted(maps.skill_hashes(payload)) == sorted(real_sources)


def test_block_z_for_old_entries():
    # An entry that origin/main wrote has no block_z fields. A --modules run adds the ones that need
    # no GPU, from the stored search scores, and leaves every stored number as it is.
    sources = synthetic.SOURCES
    plain = json.loads(json.dumps(maps.build_entry("org/toy", synthetic.engine_result(False), sources, synthetic.ROW)))
    new_keys = ["block_z", "block_z_heldout", "block_z_windows", "block_z_disjoint_windows", "block_z_disjoint",
                "block_z_disjoint_heldout"]
    old = json.loads(json.dumps(plain))
    for name in old["layer_maps"]:
        for key in new_keys:
            del old["layer_maps"][name][key]
    before = json.dumps(old, sort_keys=True)
    state = {"increases": maps.module_increases(synthetic.engine_result(True)), "engine_seconds": 1.0, "calls": 1}
    out = maps.add_module_fields(old, state, sources, None)
    assert json.dumps(old, sort_keys=True) == before, "the entry that was read is not changed"
    assert out["layer_maps"], out
    for name in out["layer_maps"]:
        lm = out["layer_maps"][name]
        want = plain["layer_maps"][name]
        for key in old["layer_maps"][name]:
            assert lm[key] == old["layer_maps"][name][key], (name, key)
        n_blocks = len(lm["blocks"])
        assert lm["block_z_windows"] == want["block_z_windows"], name
        assert lm["block_z_disjoint_windows"] == want["block_z_disjoint_windows"], name
        # from the stored scores, which are rounded to 5 places: equal to the first write within that
        for key in ["block_z", "block_z_disjoint"]:
            assert len(lm[key]) == n_blocks
            for got, first in zip(lm[key], want[key]):
                assert (got is None) == (first is None) and (got is None or abs(got - first) < 0.01), (name, key)
        assert lm["block_z"] == maps.block_z(lm["scores"], lm["blocks"])
        assert lm["block_z_heldout"] == [None] * n_blocks and lm["block_z_disjoint_heldout"] == [None] * n_blocks
        assert "held-out" in lm["block_z_note"]
    # an entry that has block_z keeps its layer maps as they are
    out = maps.add_module_fields(plain, state, sources, None)
    assert out["layer_maps"] == plain["layer_maps"]


def test_merge_keeps_no_stale_numbers():
    # A set with no base score in a later call has no module numbers at all: the numbers of the
    # earlier calls must not stay under its name.
    first = maps.module_increases(module_call([0, 1]))
    second = maps.module_increases(module_call([2, 3]))
    scored = [k for k in sorted(second["attn"]) if first["attn"][k] is not None and second["attn"][k] is not None]
    key, other = scored[0], scored[1]
    second["attn"][key] = None
    merged = maps.merge_increases(maps.merge_increases(None, first), second)
    assert merged["attn"][key] is None and merged["layers"] == [0, 1, 2, 3]
    assert merged["attn"][other][:4] == [first["attn"][other][0], first["attn"][other][1],
                                         second["attn"][other][2], second["attn"][other][3]]
    assert merged["mlp"][key] is not None


def test_maps_many_failure_memory():
    codes = {"a": 0, "b": 1, "c": maps.OLD_ENGINE_EXIT, "d": 0, "e": 0}

    def command(key, modules):
        return [sys.executable, "-c", "import sys; print('run " + key + "'); sys.exit(" + str(codes[key]) + ")"]
    with tempfile.TemporaryDirectory() as log_dir:
        # nothing remembered yet
        assert maps_many.drop_failed(["a", "b", "c"], log_dir, True, False) == (["a", "b", "c"], [])
        failed, not_started, old_engine = maps_many.run_all(["a", "b", "c", "d", "e"], 1, True, log_dir, command, 0.05)
        # c found an engine without the module call: d and e are not started
        assert failed == ["b", "c"] and not_started == ["d", "e"] and old_engine is True
        for key in ["a", "b", "c"]:
            with open(os.path.join(log_dir, key + ".log")) as f:
                assert f.read().strip() == "run " + key
        assert not os.path.exists(os.path.join(log_dir, "d.log"))
        # only b is remembered: c says nothing about the model
        marks = sorted(n for n in os.listdir(log_dir) if n.endswith(".modules-failed"))
        assert marks == ["b.modules-failed"], marks
        assert read_json(maps_many.failed_path(log_dir, "b"))["exit"] == 1
        assert maps_many.drop_failed(["a", "b", "c", "d"], log_dir, True, False) == (["a", "c", "d"], ["b"])
        # --retry-failed starts it again; a plain run is never held back
        assert maps_many.drop_failed(["a", "b"], log_dir, True, True) == (["a", "b"], [])
        assert maps_many.drop_failed(["a", "b"], log_dir, False, False) == (["a", "b"], [])
        # a run stopped by a signal is not a failure of the model
        maps_many.note_exit(log_dir, "a", -2, True)
        assert not os.path.exists(maps_many.failed_path(log_dir, "a"))
        # A run that left a model out ends with a failure code, so "0 models to map" after a bad
        # deploy is not read as "all done".
        assert maps_many.end_code([], ["b"], False) == 1 and maps_many.end_code(["a"], [], False) == 1
        assert maps_many.end_code([], [], False) == 0
        assert maps_many.end_code(["a"], ["b"], True) == maps.OLD_ENGINE_EXIT
        # --layers-per-call goes to maps.py, with --modules only
        assert maps_many.maps_command("k", True, 4)[-3:] == ["--modules", "--layers-per-call", "4"]
        assert maps_many.maps_command("k", True)[-1] == "--modules" and maps_many.maps_command("k", True, 0)[-1] == "--modules"
        assert "--layers-per-call" not in maps_many.maps_command("k", False, 4)
        # a good run forgets the failure
        codes["b"] = 0
        failed, not_started, old_engine = maps_many.run_all(["b"], 1, True, log_dir, command, 0.05)
        assert failed == [] and not_started == [] and old_engine is False
        assert not os.path.exists(maps_many.failed_path(log_dir, "b"))
    with tempfile.TemporaryDirectory() as log_dir:
        # without --modules nothing is remembered, and every model is started
        codes["b"] = 1
        codes["c"] = 1
        failed, not_started, old_engine = maps_many.run_all(["a", "b", "c", "d"], 2, False, log_dir, command, 0.05)
        assert sorted(failed) == ["b", "c"] and not_started == [] and old_engine is False
        assert sorted(os.listdir(log_dir)) == ["a.log", "b.log", "c.log", "d.log"]


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):  # the progress lines of the code under test
        test_entry_done()
        test_write_entry()
        test_plan_module_calls()
        test_measure_modules_saves_each_call()
        test_old_engine()
        test_plain_entry_fits_and_added_fields()
        test_block_z_for_old_entries()
        test_merge_keeps_no_stale_numbers()
        test_maps_many_failure_memory()
    print("ok")
