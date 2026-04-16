"""preset registry shape and smoke run.

checks that every preset in the registry has the fields the CLI relies on,
and that the `smoke` preset executes end-to-end through canonical_study
without raising."""
from spinglass.experiments.budget import Budget
from spinglass.experiments.presets import PRESETS, get_preset, list_presets
from spinglass.experiments.studies import canonical_study


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


def test_list_presets_nonempty_and_sorted():
    names = list_presets()
    assert len(names) >= 1
    assert names == sorted(names)
    assert "smoke" in names


def test_every_preset_has_required_keys():
    for name in list_presets():
        preset = get_preset(name)
        missing = REQUIRED_KEYS - set(preset)
        assert not missing, f"preset {name} missing keys: {missing}"
        assert isinstance(preset["betas"], list) and len(preset["betas"]) >= 2
        assert preset["n_disorders"] >= 1
        assert preset["n_chains"] >= 1
        assert isinstance(preset["budget"], Budget)


def test_get_preset_is_a_copy_not_a_reference():
    preset = get_preset("smoke")
    preset["mutated"] = True
    assert "mutated" not in get_preset("smoke")


def test_get_preset_raises_on_unknown_name():
    try:
        get_preset("definitely_not_a_real_preset")
    except KeyError:
        pass
    else:
        assert False, "get_preset should raise KeyError on miss"


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
    assert set(panels.keys()) == {
        ("discrete", "sampling"),
        ("discrete", "optimization"),
        ("relaxed", "sampling"),
        ("relaxed", "optimization"),
    }
    # every panel should have at least one grouped row
    for key, panel in panels.items():
        assert len(panel["grouped"]) >= 1, f"empty grouped for {key}"


if __name__ == "__main__":
    test_list_presets_nonempty_and_sorted()
    test_every_preset_has_required_keys()
    test_get_preset_is_a_copy_not_a_reference()
    test_get_preset_raises_on_unknown_name()
    test_smoke_preset_runs_end_to_end()
    print("test_presets OK")
