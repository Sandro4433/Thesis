from pathlib import Path
import sys
import os

def _ensure_project_on_syspath() -> None:
    project_dir = Path(__file__).resolve().parent  # .../python
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

def main() -> None:
    _ensure_project_on_syspath()

    # Optional: fail early if no key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    from Vision_Module.Vision_Main import main as vision_main
    from Communication_Module.API_test import main as api_main

    vision_main()  # writes File_Exchange/positions.json + llm_input.json
    api_main()     # reads File_Exchange/llm_input.json and starts interactive loop

if __name__ == "__main__":
    main()