from pathlib import Path
import sys


#This is a github repo test

def _ensure_project_on_syspath() -> None:
    project_dir = Path(__file__).resolve().parent  # .../python
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))


def main() -> None:
    _ensure_project_on_syspath()

    # API_Main handles mode selection, vision (if needed), LLM session,
    # changes application, and config saving — all in one entry point.
    from Communication_Module.API_Main import main as api_main
    api_main()

   


if __name__ == "__main__":
    main()