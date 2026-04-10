from pathlib import Path
import sys


#This is a github repo test

def _ensure_project_on_syspath() -> None:
    project_dir = Path(__file__).resolve().parent  # .../python
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))


def main() -> None:
    _ensure_project_on_syspath()

    # session_handler orchestrates the pipeline (vision, config, LLM, execution).
    from session_handler import main as session_main
    session_main()

   


if __name__ == "__main__":
    main()