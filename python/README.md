# Robot Configurator

LLM-driven workspace configurator and motion planner for the Franka Emika robot arm. Enables operators to describe workspace changes in natural language; the system translates instructions into validated PDDL-compatible configurations and pick-and-place sequences.

## Architecture

```
src/robot_configurator/
├── core/
│   ├── config.py          # Settings loaded from environment variables
│   └── paths.py           # Path constants and atomic file I/O helpers
├── communication/
│   ├── api_main.py        # Main LLM conversation loop
│   ├── prompts.py         # System prompts for reconfig and motion modes
│   ├── ambiguity_detection.py
│   ├── block_parsing.py   # Parse fenced JSON blocks from LLM responses
│   ├── capacity_tools.py  # Deterministic capacity checks (tool use)
│   ├── change_management.py
│   ├── scene_helpers.py
│   └── user_intent.py
└── configuration/
    ├── apply_config_changes.py   # Apply LLM changes to configuration.json
    ├── apply_sequence_changes.py # Apply motion sequence to state
    └── update_scene.py           # Vision-guided scene update pipeline
```

The Fast Downward PDDL planner lives in `downward/` as a git submodule.  
Vision processing lives in `Vision_Module/` (not included in this archive).  
Pipeline orchestration lives in `session_handler.py` at the project root.

## Requirements

- Python ≥ 3.9
- An OpenAI API key (GPT-4.1 or equivalent)
- Fast Downward (see [Installation](#installation))
- Ubuntu 22.04 / 24.04 recommended (tested on Ubuntu 22.04 + ROS 2 Humble)

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/robot-configurator.git
cd robot-configurator

# 2. Initialise Fast Downward as a submodule
git submodule update --init --recursive
cd downward && python build.py && cd ..

# 3. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 4. Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# 5. Copy the example env file and add your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

## Running tests

```bash
pytest                    # all unit tests
pytest -v                 # verbose
pytest --cov              # with coverage report
```

## Configuration

All tuneable parameters are set through environment variables (`.env` file).  
See `.env.example` for a full list with defaults and descriptions.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key. |
| `RC_MODEL` | `gpt-4.1` | Model to use for all LLM calls. |
| `RC_MAX_TOOL_ROUNDS` | `3` | Max tool-call rounds per LLM call. |
| `RC_POSITION_MATCH_THRESHOLD_M` | `0.040` | XY threshold (m) for part auto-matching. |
| `DOWNWARD_PATH` | `downward/fast-downward.py` | Path to the Fast Downward entry point. |

## Branching strategy

| Branch | Purpose |
|---|---|
| `main` | Stable releases only |
| `refactored-cleanup` | Active development (current) |
| `Priority_New` | Priority handling experiments |

## Citing

If you use this software in academic work, please cite:

> Gabl, S. (2026). *Natural-language-driven workspace configuration for robot manipulation*. Master's Thesis, [University].

## License

See `LICENSE` for details.
