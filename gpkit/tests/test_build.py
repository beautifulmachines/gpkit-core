"Tests for gpkit.util.build"

import os

import gpkit.util.build as build_module


def test_gpkitsolvers_empty_string_yields_no_solvers(monkeypatch, tmp_path):
    """GPKITSOLVERS="" must mean "no solvers", not [""].

    "".split(", ") is [""], not [] -- installed_solvers must stay empty,
    since downstream code checks it for truthiness.
    """
    for cls in (build_module.MosekCLI, build_module.MosekConif, build_module.CVXopt):
        monkeypatch.setattr(cls, "look", lambda _self: None)
    monkeypatch.setenv("GPKITSOLVERS", "")

    gpkit_dir = os.path.dirname(os.path.dirname(os.path.abspath(build_module.__file__)))
    real_chdir = os.chdir

    def redirected_chdir(path):
        # Keep build()'s file writes inside tmp_path instead of the real
        # package directory's env/ folder.
        real_chdir(str(tmp_path) if path == gpkit_dir else path)

    monkeypatch.setattr(build_module.os, "chdir", redirected_chdir)

    build_module.build()

    assert build_module.settings["installed_solvers"] == []
