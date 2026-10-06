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
