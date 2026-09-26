"""Every simulator module imports without Timeloop, and the port left no legacy hooks behind."""
import importlib
import os
import pathlib
import pkgutil
import re
import subprocess
import sys

import pytest

import llmforge.hw.rdxe
import llmforge.hw.timeloop

PACKAGE = pathlib.Path(llmforge.hw.timeloop.__file__).resolve().parents[2]   # src/llmforge

MODULES = sorted(
    {m.name for pkg in (llmforge.hw.timeloop, llmforge.hw.rdxe)
     for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")}
    | {"llmforge.evaluators.hw_timeloop", "llmforge.evaluators.hw_rdxe"})

OWNED = [PACKAGE / "hw" / "timeloop", PACKAGE / "hw" / "rdxe",
         PACKAGE / "evaluators" / "hw_timeloop.py", PACKAGE / "evaluators" / "hw_rdxe.py"]
LEGACY = re.compile(r"rDXE_sim|nsga_search|hw_eval/|sys\.path\.insert|os\.chdir|run_exp_hw|"
                    r"search_space import|remote_trainer|utils\.parse_timeloop_stats")
_PATTERNS = [r"FinFET", r"TSMC", r"/home/", r"/Users/", r"TOPS/W", r"silicon"]
# Author, affiliation and venue patterns stay outside the repository, since listing them would
# identify the authors. They live in the git-ignored file below, which LLMFORGE_IDENTITY_PATTERNS
# overrides, and the anonymity tests fold them in when it is present.
_LOCAL = PACKAGE.parents[1] / "configs" / "local" / "identity_patterns.txt"
_EXTRA = os.environ.get("LLMFORGE_IDENTITY_PATTERNS", str(_LOCAL))
if _EXTRA and pathlib.Path(_EXTRA).is_file():
    _PATTERNS += [l.strip() for l in pathlib.Path(_EXTRA).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
IDENTIFYING = re.compile("|".join(_PATTERNS), re.IGNORECASE)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_modules_import_with_timeloopfe_blocked():
    # A None entry in sys.modules makes `import timeloopfe` raise, so any eager import fails here
    # even on machines where Timeloop is installed.
    code = ("import importlib, sys\n"
            "sys.modules['timeloopfe'] = None\n"
            f"for name in {MODULES!r}:\n"
            "    importlib.import_module(name)\n")
    src = str(PACKAGE.parent)
    path = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": path})


def _owned_files():
    for root in OWNED:
        if root.is_file():
            yield root
        else:
            yield from sorted(p for p in root.rglob("*") if p.suffix in {".py", ".yaml", ".md"})


def test_no_legacy_hooks_or_identifying_strings():
    bad = []
    for f in _owned_files():
        text = f.read_text()
        for rx in (LEGACY, IDENTIFYING):
            bad += [f"{f.relative_to(PACKAGE)}: {m.group(0)}" for m in rx.finditer(text)]
    assert not bad, bad
