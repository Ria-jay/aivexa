from aivexa.targets.config import (
    TargetConfig,
    TargetRegistry,
)


def test_target_registry_persists_configuration(tmp_path):
    path = tmp_path / "targets.json"

    registry = TargetRegistry(str(path))

    config = TargetConfig(
        name="test-gemini",
        target_type="gemini",
        endpoint="https://example.test",
        model="test-model",
        api_key_env="GEMINI_API_KEY",
        capabilities={
            "rag": True,
            "tools": True,
        },
    )

    registry.save(config)

    loaded = registry.get("test-gemini")

    assert loaded.name == "test-gemini"
    assert loaded.target_type == "gemini"
    assert loaded.model == "test-model"
    assert loaded.capabilities["tools"] is True


def test_target_config_validation_rejects_invalid_configuration():
    import pytest

    with pytest.raises(ValueError):
        TargetConfig(
            name="",
            target_type="gemini",
            endpoint="https://example.test",
        ).validate()

    with pytest.raises(ValueError):
        TargetConfig(
            name="x",
            target_type="gemini",
            endpoint="https://example.test",
            timeout=0,
        ).validate()


def test_target_registry_rejects_invalid_configuration(tmp_path):
    import pytest

    registry = TargetRegistry(str(tmp_path / "targets.json"))

    with pytest.raises(ValueError):
        registry.save(
            TargetConfig(
                name="bad",
                target_type="",
                endpoint="",
            )
        )
