# Changelog

All notable changes to this project are documented here.

## [Unreleased] — refactored-cleanup branch

### Added
- `pyproject.toml` — installable package with `pip install -e ".[dev]"`
- `src/` layout with single top-level package `robot_configurator`
- `core/config.py` — centralised `Settings` class; all magic strings and thresholds now come from environment variables
- `core/paths.py` — atomic `save_atomic()`, versioned `save_to_memory()`, `empty_state()`, `parent_of_slot()` helpers
- `.env.example` — documents all required and optional environment variables
- `.gitignore` — excludes `__pycache__`, `.env`, `downward/`, runtime outputs
- `.github/workflows/ci.yml` — GitHub Actions pipeline (pytest + ruff, Python 3.9 and 3.11)
- `README.md` — installation guide, architecture overview, configuration reference
- `tests/` — 40+ unit tests covering `block_parsing`, `change_management`, `scene_helpers`, `apply_config_changes`, `apply_sequence_changes`

### Changed
- All module filenames renamed to `snake_case` (PEP 8):
  - `API_Main.py` → `communication/api_main.py`
  - `Apply_Config_Changes.py` → `configuration/apply_config_changes.py`
  - `Apply_Sequence_Changes.py` → `configuration/apply_sequence_changes.py`
  - `Update_Scene.py` → `configuration/update_scene.py`
- All `sys.path` bootstrap blocks removed — replaced by proper packaging
- All `from Communication_Module.X import` → `from robot_configurator.communication.x import`
- All `from Configuration_Module.X import` → `from robot_configurator.configuration.x import`
- `import paths` → `from robot_configurator.core.paths import ...`
- Hardcoded `MODEL = "gpt-4.1"` → `settings.model` (from env var `RC_MODEL`)
- Hardcoded `MAX_TOOL_ROUNDS = 3` → `settings.max_tool_rounds`
- Hardcoded `POSITION_MATCH_THRESHOLD_M = 0.040` → `settings.position_match_threshold_m`
- All `print(f"  [Tool: ...]")` debug statements → `logging.debug(...)` calls
- `downward/` moved out of the repository and registered as a git submodule

### Removed
- `Communication_Module/` and `Configuration_Module/` directory names
- `__pycache__/` and `.pyc` files from version control
- `rules.txt` from inside the Python package (content folded into `prompts.py`)
