# Changelog

All notable changes to this project are documented here.

## [Unreleased] — Priority_New branch

### Fixed
- Removed hardcoded absolute path `/home/hv/...` from `Vision_Module/config.py`;
  `FAST_DOWNWARD_PATH` now resolves relative to the project directory and can
  be overridden with the `DOWNWARD_PATH` environment variable
- `downward/` registered as a git submodule (was previously committed as a raw
  609 MB directory)

### Changed
- All `from paths import …` / `import paths` calls in `Vision_Module/`,
  `Execution_Module/`, and orchestration files updated to
  `from robot_configurator.core.paths import …`
- `Main.py` simplified — bootstrap `sys.path` block removed (packaging handles
  path resolution)

### Removed
- `Old/Vision_test.py` — superseded by `Vision_Module/pipeline.py`
- Runtime images (`image_original.png`, `image_undistorted.png`) removed from
  version control (now covered by `.gitignore`)
- Session state files (`Memory/*.json`) removed from version control

---

## [0.1.0] — refactored-cleanup branch

### Added
- `pyproject.toml` — installable package with `pip install -e ".[dev]"`
- `src/` layout with single top-level package `robot_configurator`
- `core/config.py` — centralised `Settings` class; all magic strings and
  thresholds now come from environment variables
- `core/paths.py` — atomic `save_atomic()`, versioned `save_to_memory()`,
  `empty_state()`, `parent_of_slot()` helpers
- `.env.example` — documents all required and optional environment variables
- `.gitignore` — excludes `__pycache__`, `.env`, `downward/`, runtime outputs
- `.gitmodules` — registers Fast Downward as a git submodule
- `.github/workflows/ci.yml` — GitHub Actions pipeline (pytest + ruff,
  Python 3.9 and 3.11)
- `README.md` — installation guide, architecture overview, configuration
  reference
- `tests/` — 40+ unit tests covering `block_parsing`, `change_management`,
  `scene_helpers`, `apply_config_changes`, `apply_sequence_changes`

### Changed
- All module filenames renamed to `snake_case` (PEP 8)
- All `sys.path` bootstrap blocks removed — replaced by proper packaging
- All `from Communication_Module.X import` → `from robot_configurator.communication.x import`
- All `from Configuration_Module.X import` → `from robot_configurator.configuration.x import`
- Hardcoded model/threshold values → environment-variable-backed `Settings`
- All `print(f"  [Tool: ...]")` debug statements → `logging.debug(...)` calls

### Removed
- `Communication_Module/` and `Configuration_Module/` directory names
- `__pycache__/` and `.pyc` files from version control
- `rules.txt` from inside the Python package (content folded into `prompts.py`)
