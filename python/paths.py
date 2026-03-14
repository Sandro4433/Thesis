"""Shared, stable filesystem paths for the project.

All paths are defined relative to the project root directory (the `python/` folder).
Use these constants instead of `os.path.abspath(...)` so code works regardless of
the current working directory.
"""

from pathlib import Path


# Project root = the folder containing this file (python/)
PROJECT_DIR = Path(__file__).resolve().parent


# Shared exchange folder for cross-module data
FILE_EXCHANGE_DIR = PROJECT_DIR / "File_Exchange"


# Common files
CONFIGURATION_JSON    = FILE_EXCHANGE_DIR / "configuration.json"
LLM_INPUT_JSON        = FILE_EXCHANGE_DIR / "llm_input.json"
LLM_RESPONSE_JSON     = FILE_EXCHANGE_DIR / "llm_response.json"
POSITIONS_FIXED_JSONL = FILE_EXCHANGE_DIR / "positions_fixed.jsonl"