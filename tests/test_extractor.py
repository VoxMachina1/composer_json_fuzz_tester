import sys
import os

# Make fuzz_tester.fuzz_tester importable from the project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_FUZZ_DIR = os.path.dirname(_HERE)          # fuzz_tester/
_PROJECT_ROOT = os.path.dirname(_FUZZ_DIR)  # strategy_viewer/
sys.path.insert(0, _FUZZ_DIR)               # so `import fuzz_tester` resolves

# Also set up strategy_engine/src path exactly as fuzz_tester.py does (lines 29-30)
import pathlib
_STRATEGY_ENGINE_SRC = pathlib.Path(_FUZZ_DIR) / "strategy_engine" / "src"
sys.path.insert(0, str(_STRATEGY_ENGINE_SRC))

from fuzz_tester import extract_conditions_from_tree

FIXTURE_DIR = pathlib.Path(_FUZZ_DIR) / "pathfinder"


def _load_tree(filename):
    import json
    with open(FIXTURE_DIR / filename) as f:
        return json.load(f)


def test_any_all_example():
    tree = _load_tree("any_all_example.json")
    conditions = extract_conditions_from_tree(tree)
    assert len(conditions) == 6, f"Expected 6 conditions, got {len(conditions)}"
    for cond in conditions:
        assert cond.get("children_endpoints"), \
            f"Condition {cond['id']} has empty children_endpoints: {cond}"
        assert cond.get("category") != "unknown", \
            f"Condition {cond['id']} has unknown category: {cond}"
    return True


def test_bestsignals3():
    tree = _load_tree("bestsignals3.json")
    conditions = extract_conditions_from_tree(tree)
    assert len(conditions) == 37, f"Expected 37 conditions, got {len(conditions)}"
    for cond in conditions:
        assert cond.get("category") != "unknown", \
            f"Condition {cond['id']} has unknown category: {cond}"
    return True


if __name__ == "__main__":
    tests = [test_any_all_example, test_bestsignals3]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failures.append(t.__name__)
    if failures:
        print(f"\n{len(failures)} test(s) failed: {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed.")
        sys.exit(0)
