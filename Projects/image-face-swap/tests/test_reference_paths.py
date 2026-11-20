"""Reference-directory migration must not break automatic asset discovery."""

import pickle

import pytest

from facework import server


@pytest.fixture
def repository(tmp_path, monkeypatch):
    workspace = tmp_path / "Projects" / "image-face-swap"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(server, "WORKSPACE", workspace)
    monkeypatch.delenv("FACEWORK_ASSETS_DIR", raising=False)
    monkeypatch.delenv("FACEWORK_DATA_DIR", raising=False)
    return tmp_path


def reference_fixture(root, iteration=42):
    assets = root / "DeepFaceLab" / "facelib"
    assets.mkdir(parents=True)
    for name in ("S3FD.npy", "2DFAN.npy"):
        (assets / name).write_bytes(b"placeholder")
    model = root / "AI face swap " / "AI" / "AI" / "workspace" / "model"
    model.mkdir(parents=True)
    prefix = "Example_SAEHD"
    (model / (prefix + "_data.dat")).write_bytes(pickle.dumps({
        "options": {"archi": "df-d", "resolution": 224, "face_type": "f"},
        "iter": iteration,
    }, protocol=4))
    for part in ("encoder", "inter", "decoder_src", "decoder_dst"):
        (model / (prefix + "_" + part + ".npy")).write_bytes(b"placeholder")
    return assets, model


@pytest.mark.parametrize("moved", [False, True])
def test_reference_assets_and_model_work_before_and_after_move(repository, moved):
    root = repository / "LocalReferences" if moved else repository
    assets, model = reference_fixture(root)
    runtime = server.Runtime(device="cpu")
    assert runtime.asset_dir == assets.resolve()
    assert runtime.status()["assets_ready"]
    assert runtime.issues == []
    assert [item["directory"] for item in runtime.models.values()] == [str(model.resolve())]
    assert not runtime.registry_path.exists(), "Auto discovery must not persist stale absolute paths"


def test_new_reference_location_takes_priority_when_both_exist(repository):
    reference_fixture(repository, iteration=1)
    assets, model = reference_fixture(repository / "LocalReferences", iteration=2)
    runtime = server.Runtime(device="cpu")
    assert runtime.asset_dir == assets.resolve()
    assert len(runtime.models) == 1
    descriptor = next(iter(runtime.models.values()))
    assert descriptor["directory"] == str(model.resolve())
    assert descriptor["iteration"] == 2


def test_workspace_assets_take_priority_over_reference_sources(repository):
    reference_fixture(repository / "LocalReferences")
    bundled = server.WORKSPACE / "assets" / "facelib"
    bundled.mkdir(parents=True)
    runtime = server.Runtime(device="cpu")
    assert runtime.asset_dir == bundled.resolve()


def test_explicit_and_environment_assets_override_discovery(repository, monkeypatch):
    reference_fixture(repository / "LocalReferences")
    environment_assets = repository / "custom-environment"
    explicit_assets = repository / "custom-explicit"
    monkeypatch.setenv("FACEWORK_ASSETS_DIR", str(environment_assets))
    assert server.Runtime(device="cpu").asset_dir == environment_assets.resolve()
    assert server.Runtime(device="cpu", asset_dir=explicit_assets).asset_dir == explicit_assets.resolve()


def test_missing_references_show_status_issue_without_crashing(repository):
    runtime = server.Runtime(device="cpu")
    assert not runtime.status()["assets_ready"]
    assert runtime.status()["issues"]
    assert runtime.models == {}
