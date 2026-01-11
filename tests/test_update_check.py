from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

import pdf2tree.core as core


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:  # noqa: D401
        return self._data

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


def test_update_check_skipped_in_test_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("PDF2TREE_TEST_MODE", "1")

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("Network call should not happen in test mode")

    monkeypatch.setattr(core.urllib.request, "urlopen", _boom)

    core.maybe_print_new_version_notice(program_name="pdf2tree")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_update_check_silent_on_network_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # In pytest PYTEST_CURRENT_TEST is set; force disable test mode.
    monkeypatch.setenv("PDF2TREE_TEST_MODE", "0")

    def _raise(*args: object, **kwargs: object) -> object:
        raise OSError("offline")

    monkeypatch.setattr(core.urllib.request, "urlopen", _raise)

    core.maybe_print_new_version_notice(program_name="pdf2tree")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_update_check_prints_notice_when_latest_is_greater(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("PDF2TREE_TEST_MODE", "0")

    # Latest greater than current (project version is 0.0.6).
    dummy = _DummyResponse({"tag_name": "v0.0.7"})

    def _ok(req: object, timeout: float | None = None) -> _DummyResponse:
        assert timeout == core.UPDATE_CHECK_TIMEOUT_SECONDS
        return dummy

    monkeypatch.setattr(core.urllib.request, "urlopen", _ok)

    core.maybe_print_new_version_notice(program_name="pdf2tree")
    captured = capsys.readouterr()
    assert (
        captured.out
        == f"A new version of pdf2tree is available: current {core.program_version()}, latest 0.0.7. To upgrade, run: pdf2tree --upgrade\n"
    )


def test_update_check_silent_when_latest_not_greater(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("PDF2TREE_TEST_MODE", "0")

    dummy = _DummyResponse({"tag_name": f"v{core.program_version()}"})

    def _ok(req: object, timeout: float | None = None) -> _DummyResponse:
        return dummy

    monkeypatch.setattr(core.urllib.request, "urlopen", _ok)

    core.maybe_print_new_version_notice(program_name="pdf2tree")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_upgrade_flag_runs_pip_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDF2TREE_TEST_MODE", "1")
    calls: dict[str, Any] = {}

    def _fake_run(cmd: list[str]) -> SimpleNamespace:
        calls["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(core.subprocess, "run", _fake_run)
    monkeypatch.setattr(core.sys, "argv", ["pdf2tree", "--upgrade"])

    assert core.main() == 0
    cmd = calls.get("cmd")
    assert isinstance(cmd, list)
    assert cmd[:4] == [core.sys.executable, "-m", "pip", "install"]
    assert "--upgrade" in cmd
    assert cmd[-1] == "pdf2tree"
