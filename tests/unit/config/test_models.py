from simulation_harness.config.models import GenerationConfig, StartupConfig


def test_generation_config_scenario_defaults() -> None:
    cfg = GenerationConfig()
    assert cfg.scenarios_enabled is True
    assert cfg.scenarios_count == 5
    assert cfg.scenarios.temperature == 0.4
    assert cfg.scenarios.max_tokens == 3000


def test_startup_autostart_enabled_defaults_false() -> None:
    assert StartupConfig().autostart_enabled is False


def test_startup_autostart_enabled_accepts_true() -> None:
    assert StartupConfig(autostart_enabled=True).autostart_enabled is True


def test_ablation_spec_only_is_accepted() -> None:
    """skill=False, state=False is the `spec_only` reference condition."""
    from simulation_harness.config.models import AblationConfig

    abl = AblationConfig(skill=False, state=False)
    assert (abl.skill, abl.state) == (False, False)


def test_ablation_refuses_state_off_with_skill_on() -> None:
    """skill=True, state=False is not a condition we run: a skill whose contracts tell the
    simulator to read and write a store it does not have. Refused at config load, so a harness
    never starts serving it."""
    import pytest
    from pydantic import ValidationError

    from simulation_harness.config.models import AblationConfig

    with pytest.raises(ValidationError, match="HARNESS_ABLATION_STATE=false"):
        AblationConfig(skill=True, state=False)
