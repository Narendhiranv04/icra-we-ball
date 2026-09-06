from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mujoco_scenes.baselines.vilain_tamp.config import BaselineConfig, Domain
from mujoco_scenes.baselines.vilain_tamp.fm import FMTransportResponse, RecordedFMClient
from mujoco_scenes.baselines.vilain_tamp.live_fm import LiveModelClients
from mujoco_scenes.baselines.vilain_tamp.runner import RunOptions
from mujoco_scenes.baselines.vilain_tamp.runtime import (
    RuntimeCompositionError,
    _neutral_entity_types,
    build_live_components,
)
from mujoco_scenes.run_vilain_tamp_baseline import main


class NullTransport:
    def complete(self, request):
        return FMTransportResponse("unused", "unused", request.model, None, {})


def _inputs(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config = BaselineConfig.from_yaml(root / "configs/qwen_only.yaml")
    options = RunOptions(
        domain=Domain.KITCHEN,
        variant="K1",
        observation_mode=config.observation_mode,
        model_condition=config.model_condition,
        output_directory=tmp_path,
        cp_limit=3,
        fast_downward_path=config.external_tools.fast_downward,
        val_path=config.external_tools.val,
    )
    return config, options


def test_production_factory_composes_planning_only_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mujoco_scenes.baselines.vilain_tamp.runtime as runtime

    config, options = _inputs(tmp_path)
    protocol = SimpleNamespace(
        domain=Domain.KITCHEN,
        observation_mode=config.observation_mode,
        output_root=tmp_path / "observations",
    )
    scene = SimpleNamespace(model=object(), data=object())
    monkeypatch.setattr(
        runtime,
        "create_live_observation_runtime",
        lambda **kwargs: SimpleNamespace(scene=scene, protocol=protocol),
    )
    monkeypatch.setattr(
        runtime,
        "create_live_refinement_runtime",
        lambda *args, **kwargs: SimpleNamespace(
            refiner=object(), planning_scene_factory=lambda: object()
        ),
    )
    monkeypatch.setattr(runtime, "FastDownwardPlanner", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "TAMPAttemptRunner", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        runtime,
        "CorrectivePlanningLoop",
        lambda **kwargs: SimpleNamespace(max_corrections=kwargs["max_corrections"]),
    )
    transport = NullTransport()
    clients = LiveModelClients(
        RecordedFMClient(transport),
        RecordedFMClient(transport),
        "qwen35-9b",
        None,
        "qwen35-9b",
        None,
    )

    components = build_live_components(
        config,
        options,
        served_model_id="qwen35-9b",
        model_clients=clients,
    )

    assert components.observation is protocol
    assert components.execution is None
    assert components.interpreter.models.object_estimator_model == "qwen35-9b"
    assert components.interpreter.models.reasoning_model == "qwen35-9b"
    assert components.corrective_planning.max_corrections == 3

    with pytest.raises(RuntimeCompositionError, match="planning-only"):
        build_live_components(
            config,
            RunOptions(**{**options.__dict__, "execute": True}),
            served_model_id="qwen35-9b",
            model_clients=clients,
        )


def test_cli_help_is_side_effect_free(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    assert "--live" in capsys.readouterr().out


@pytest.mark.parametrize(
    "domain,name,expected",
    [
        ("kitchen", "s1i_oversized_spoon", "utensil"),
        ("kitchen", "ab3_deep_bowl", "vessel"),
        ("kitchen", "s1i_compact_kettle", "source"),
        ("living_room", "a2_remote_payload", "remote"),
        ("workshop", "workshop_medium_phillips_screw", "fastener"),
    ],
)
def test_neutral_body_taxonomy_restricts_candidate_types(
    domain: str, name: str, expected: str
) -> None:
    assert expected in _neutral_entity_types(name, movable=True, domain_key=domain)
