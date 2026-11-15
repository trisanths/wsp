# CPU test of maps.main() from the command line arguments to the entry file. The engine is the real
# engine_app.EngineBase code on the toy decoder of test_skill_maps_engine.py; only the Modal handle
# (Cls.from_name(...)(...).skill_maps.remote) is replaced by a direct call, and the revision pin and
# the skill files by small local values. Nothing here talks to Modal or to the network.
#   .venv/bin/python mine/tests/test_maps_main.py
import contextlib
import io
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import maps  # noqa: E402
import test_skill_maps_engine as toy  # noqa: E402

KEY = "Toy__Model"
SOURCES = {"law": "toy/law train: toy law text", maps.REFERENCE: "toy/general train: toy general text"}
DETAILS = {"law": {"dataset": "toy/law train", "description": "toy law text", "pool": None, "reported_benchmark": False},
           maps.REFERENCE: {"dataset": "toy/general train", "description": "toy general text", "pool": None,
                            "reported_benchmark": False}}


class Remote:
    # stands for a Modal method handle: .remote(...) runs the engine method in this process
    def __init__(self, world):
        self.world = world

    def remote(self, *args, **kwargs):
        world = self.world
        world.calls.append(dict(kwargs))
        if world.fail_at is not None and len(world.calls) == world.fail_at:
            raise RuntimeError("the call reached its timeout")
        if world.old_engine:
            # the signature of skill_maps on main: an unknown keyword is a TypeError
            def skill_maps(payload, routing_items=64):
                return world.engine.skill_maps(payload, routing_items)
            res = skill_maps(*args, **kwargs)
        else:
            res = world.engine.skill_maps(*args, **kwargs)
        if world.after_call is not None:
            world.after_call()
        return res


class World:
    # one temp repo state: entry and log directories, a zoo with one model, and the toy engine
    def __init__(self, tmp):
        self.engine = toy.CpuEngine()
        self.calls = []
        self.fail_at = None
        self.old_engine = False
        self.after_call = None
        self.revision = "rev1"
        self.out_dir = os.path.join(tmp, "pool_entries")
        self.log_dir = os.path.join(tmp, "pool_logs")
        self.zoo = os.path.join(tmp, "zoo.json")
        with open(self.zoo, "w") as f:
            json.dump({"models": [{"key": KEY, "id": "org/toy", "params_b": 0.001, "family": "toy", "license": "mit"}]}, f)
        self.entry_path = os.path.join(self.out_dir, "org__toy.json")
        self.partial_path = os.path.join(self.log_dir, "org__toy.modules-partial.json")

    def entry(self):
        with open(self.entry_path) as f:
            return json.load(f)

    def run(self, *argv):
        # maps.main() with these arguments. Returns (exit code or None, error or None, printed text).
        self.calls = []
        handle = types.SimpleNamespace(skill_maps=Remote(self))
        fake_modal = types.SimpleNamespace(Cls=types.SimpleNamespace(from_name=lambda app, gpu: lambda **params: handle))
        saved = (maps.OUT_DIR, maps.LOG_DIR, maps.ZOO, maps.load_skills, maps.load_selection_sources,
                 common.pin_revision, sys.argv, sys.modules.get("modal"))
        maps.OUT_DIR, maps.LOG_DIR, maps.ZOO = self.out_dir, self.log_dir, self.zoo
        maps.load_skills = lambda: (toy.payload(), dict(SOURCES))
        maps.load_selection_sources = lambda healdata: json.loads(json.dumps(DETAILS))
        common.pin_revision = lambda hf_id: self.revision
        sys.argv = ["maps.py", "--zoo-key", KEY] + list(argv)
        sys.modules["modal"] = fake_modal
        text = io.StringIO()
        code = None
        error = None
        toy.block_network()
        try:
            with contextlib.redirect_stdout(text):
                maps.main()
        except SystemExit as e:
            code = e.code
        except Exception as e:  # noqa: BLE001  the test looks at it
            error = e
        finally:
            toy.restore_network()
            (maps.OUT_DIR, maps.LOG_DIR, maps.ZOO, maps.load_skills, maps.load_selection_sources,
             common.pin_revision, sys.argv) = saved[:7]
            if saved[7] is None:
                del sys.modules["modal"]
            else:
                sys.modules["modal"] = saved[7]
        return code, error, text.getvalue()


def layers_asked(calls):
    out = []
    for kwargs in calls:
        out.extend(kwargs["layers"])
    return out


def test_plain_then_modules():
    with tempfile.TemporaryDirectory() as tmp:
        world = World(tmp)
        n = world.engine.info["n_layers"]
        code, error, text = world.run()
        assert code is None and error is None, (code, error, text)
        # the same call as before --modules existed: the payload and no keyword
        assert world.calls == [{}]
        plain = world.entry()
        assert "module_maps" not in plain and plain["revision"] == "rev1" and plain["layers"] == n
        assert plain["selection_sources"] == DETAILS and plain["reported_benchmark_skills"] == []
        assert not os.path.exists(world.partial_path)

        # a second plain run does nothing
        code, error, text = world.run()
        assert world.calls == [] and "exists:" in text

        # --modules keeps the plain entry and pays only for the module passes
        world.engine.row_calls = 0
        code, error, text = world.run("--modules")
        assert code is None and error is None, (code, error, text)
        for kwargs in world.calls:
            assert kwargs["modules"] is True and kwargs["plain"] is False
        assert layers_asked(world.calls) == list(range(n))
        # the base is cached in the container; 5 attention modules and 6 feed-forward modules, 4 sets
        assert world.engine.row_calls == 4 * 11, world.engine.row_calls
        full = world.entry()
        for key in plain:
            assert full[key] == plain[key], key
        assert sorted(set(full.keys()) - set(plain.keys())) == [
            "module_calls", "module_engine_seconds", "module_maps", "module_measured_at", "module_method",
            "module_parts"]
        assert full["module_calls"] == len(world.calls)
        # the same module maps as from one engine call for the whole model
        one = world.engine.skill_maps(toy.payload(), modules=True)
        want = json.loads(json.dumps(maps.build_entry("org/toy", one, SOURCES, {})))
        assert full["module_maps"] == want["module_maps"] and full["module_parts"] == want["module_parts"]
        assert not os.path.exists(world.partial_path)
        assert os.listdir(world.out_dir) == ["org__toy.json"]

        # done: a second --modules run makes no call
        code, error, text = world.run("--modules")
        assert world.calls == [] and "exists:" in text

        # a plain run that ends now must not put an entry without module maps over this one. The
        # skip rule stops it before the call; the write guard stops a run that started earlier.
        def modules_run_ends_first():
            maps.write_entry(world.entry_path, full, True)
        os.remove(world.entry_path)
        world.after_call = modules_run_ends_first
        code, error, text = world.run()
        world.after_call = None
        assert world.calls == [{}] and "kept" in text and "--force" in text, text
        assert world.entry() == full
        # --force says to replace it
        code, error, text = world.run("--force")
        assert world.calls == [{}] and "module_maps" not in world.entry()


def test_modules_from_nothing_and_a_failed_call():
    with tempfile.TemporaryDirectory() as tmp:
        world = World(tmp)
        n = world.engine.info["n_layers"]
        # no entry yet: the plain call first, then the module calls; the second module call fails
        world.fail_at = 3
        code, error, text = world.run("--modules", "--layers-per-call", "4")
        assert isinstance(error, RuntimeError) and "timeout" in str(error), (code, error, text)
        assert world.calls[0] == {} and layers_asked(world.calls[1:]) == [0, 1, 2, 3, 4, 5]
        # the plain entry and the first module call are on disk
        assert "module_maps" not in world.entry() and world.entry()["layers"] == n
        with open(world.partial_path) as f:
            saved = json.load(f)
        assert saved["increases"]["layers"] == [0, 1, 2, 3] and saved["calls"] == 1
        assert saved["stamp"] == {"model": "org/toy", "revision": "rev1", "layers": n,
                                  "skills": {"law": "law1", maps.REFERENCE: "gen1"}}

        # the next run asks only for the layers that are missing
        world.fail_at = None
        code, error, text = world.run("--modules", "--layers-per-call", "4")
        assert code is None and error is None, (code, error, text)
        assert [kwargs["layers"] for kwargs in world.calls] == [[4, 5]]
        full = world.entry()
        one = world.engine.skill_maps(toy.payload(), modules=True)
        want = json.loads(json.dumps(maps.build_entry("org/toy", one, SOURCES, {})))
        assert full["module_maps"] == want["module_maps"] and full["module_calls"] == 2
        assert not os.path.exists(world.partial_path)

        # the model has a new revision: the entry does not fit, everything is measured again
        world.revision = "rev2"
        code, error, text = world.run("--modules", "--force", "--layers-per-call", "6")
        assert code is None and error is None, (code, error, text)
        assert world.calls[0] == {} and [kwargs["layers"] for kwargs in world.calls[1:]] == [list(range(n))]
        assert world.entry()["revision"] == "rev2" and "module_maps" in world.entry()

    with tempfile.TemporaryDirectory() as tmp:
        world = World(tmp)
        world.run()
        # a plain entry of another revision is not mixed with module maps of this one
        world.revision = "rev2"
        code, error, text = world.run("--modules")
        assert code is None and error is None, (code, error, text)
        assert world.calls[0] == {} and "another revision" in text
        assert world.entry()["revision"] == "rev2" and "module_maps" in world.entry()


def test_old_engine_and_refusal():
    with tempfile.TemporaryDirectory() as tmp:
        world = World(tmp)
        world.old_engine = True
        code, error, text = world.run("--modules")
        # the deploy message and its own exit code, not a traceback
        assert code == maps.OLD_ENGINE_EXIT and error is None, (code, error, text)
        assert maps.OLD_ENGINE_MESSAGE in text and "unexpected keyword argument" in text
        # the plain call works on the old engine, and its result is kept
        assert len(world.calls) == 2 and world.calls[0] == {}
        assert "module_maps" not in world.entry()
        assert not os.path.exists(world.partial_path)
        # a plain run on the old engine is not touched by any of this
        code, error, text = world.run("--force")
        assert code is None and error is None and world.calls == [{}]

        # the plain run says that one scoring pass takes longer than a third of the call timeout
        world.old_engine = False
        entry = world.entry()
        entry["engine_seconds"] = (maps.CALL_TIMEOUT / 3 + 1) * (1 + entry["layers"])
        maps.write_entry(world.entry_path, entry, True)
        code, error, text = world.run("--modules")
        assert isinstance(code, str) and "do not fit" in code and world.calls == [], (code, error, text)
        assert "module_maps" not in world.entry()


if __name__ == "__main__":
    test_plain_then_modules()
    test_modules_from_nothing_and_a_failed_call()
    test_old_engine_and_refusal()
    print("ok")
