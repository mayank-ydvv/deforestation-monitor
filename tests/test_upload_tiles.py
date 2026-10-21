"""Offline tests for the Hugging Face publishing helper (fake API, no network)."""

from __future__ import annotations

from pathlib import Path

from training import upload_tiles


class _FakeApi:
    """Records calls; mimics huggingface_hub 2.x (no upload_large_folder)."""

    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def upload_folder(self, **kw):
        files = sorted(
            p.relative_to(kw["folder_path"]).as_posix()
            for p in Path(kw["folder_path"]).rglob("*")
            if p.is_file()
        )
        self.calls.append(("upload_folder", files, kw))


class _FakeApiV0(_FakeApi):
    """Mimics huggingface_hub 0.x, which has upload_large_folder."""

    def upload_large_folder(self, **kw):
        self.calls.append(("upload_large_folder", None, kw))


def test_publish_with_hub_2x(tmp_path):
    api = _FakeApi()
    upload_tiles.publish(api, "me/tiles", tiles_dir=tmp_path)
    names = [c[0] for c in api.calls]
    assert names == ["create_repo", "upload_folder", "upload_folder"]
    assert api.calls[0][2]["repo_type"] == "dataset" and api.calls[0][2]["exist_ok"]
    extras = api.calls[1][1]
    assert {
        "norm_stats.json",
        "README.md",
        "code/training/train.py",
        "code/pipeline/config.py",
    } <= set(extras)
    assert api.calls[2][2]["allow_patterns"] == upload_tiles.TILE_PATTERNS
    assert api.calls[2][2]["folder_path"] == tmp_path


def test_publish_prefers_large_folder_on_hub_0x(tmp_path):
    api = _FakeApiV0()
    upload_tiles.publish(api, "me/tiles", tiles_dir=tmp_path)
    assert [c[0] for c in api.calls][-1] == "upload_large_folder"
