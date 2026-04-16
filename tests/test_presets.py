# PRESETS registry sanity: required keys, copy-not-reference, unknown raises, smoke end-to-end
from spinglass.experiments.budget import Budget
from spinglass.experiments.presets import PRESETS, get_preset, list_presets
from spinglass.experiments.studies import canonical_study

# every preset must define at least these fields for canonical_study to consume
REQUIRED_KEYS = {
    "model_class",
    "model_kwargs",
    "betas",
    "n_disorders",
    "n_chains",
    "n_restarts",
    "budget",
    "description",
}

# list_presets returns a non-empty alphabetized list that always contains "smoke"
def test_list_presets_nonempty_and_sorted():
    names = list_presets()
    assert len(names) >= 1
    assert names == sorted(names)
    # "smoke" is the contract-tested baseline preset, always present
    assert "smoke" in names

# each preset must have all required keys and well-typed values
def test_every_preset_has_required_keys():
    for name in list_presets():
        preset = get_preset(name)
        missing = REQUIRED_KEYS - set(preset)
        assert not missing, f"preset {name} missing keys: {missing}"
        # need at least 2 betas for a meaningful sweep
        assert isinstance(preset["betas"], list) and len(preset["betas"]) >= 2
        assert preset["n_disorders"] >= 1
        assert preset["n_chains"] >= 1
        assert isinstance(preset["budget"], Budget)

# get_preset must hand back an independent dict so callers can mutate safely
def test_get_preset_is_a_copy_not_a_reference():
    preset = get_preset("smoke")
    preset["mutated"] = True
    # second call should be untouched by the mutation above
    assert "mutated" not in get_preset("smoke")

# unknown preset names must raise KeyError (not return None silently)
def test_get_preset_raises_on_unknown_name():
    try:
        get_preset("definitely_not_a_real_preset")
    except KeyError:
        pass
    else:
        assert False, "get_preset should raise KeyError on miss"

# smoke preset is small enough to actually run; check all 4 panels come back non-empty
def test_smoke_preset_runs_end_to_end():
    preset = get_preset("smoke")
    panels = canonical_study(
        preset["model_class"],
        preset["model_kwargs"],
        betas=preset["betas"],
        n_chains=preset["n_chains"],
        n_restarts=preset["n_restarts"],
        n_disorders=preset["n_disorders"],
        budget=preset["budget"],
    )
    # canonical_study returns the full (space, mode) cross product
    assert set(panels.keys()) == {
        ("discrete", "sampling"),
        ("discrete", "optimization"),
        ("relaxed", "sampling"),
        ("relaxed", "optimization"),
    }
    for key, panel in panels.items():
        assert len(panel["grouped"]) >= 1, f"empty grouped for {key}"

if __name__ == "__main__":
    test_list_presets_nonempty_and_sorted()
    test_every_preset_has_required_keys()
    test_get_preset_is_a_copy_not_a_reference()
    test_get_preset_raises_on_unknown_name()
    test_smoke_preset_runs_end_to_end()
    print("test_presets OK")
