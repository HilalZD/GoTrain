# Contract requests (Serena → Hilal)

Things the ML side needs from files Hilal owns. Nothing here is changed until we both agree.
Each entry: what, why, and the smallest change that would do it.

## 1. FYI: engine tests live in new files under `tests/` (no action needed)

- **What:** Serena's tests are new files only: `tests/test_engine_data.py` (M1), and later
  `tests/test_engine_train.py`, `tests/test_engine_diagnostics.py`. Existing test files are
  never edited.
- **Why:** `make test` collects `tests/` only; keeping engine tests there means one command
  runs everything. Agreed with Serena on 2026-10-31.
- **Data-dependent tests** skip themselves when the GEO-Bench files are not on disk, so the
  suite stays green on machines without data.
