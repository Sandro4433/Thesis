import sys
# Bootstrap: ensure project root is on sys.path before importing paths.
_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import paths  # ensures PROJECT_DIR is on sys.path


def main() -> None:
    # session_handler orchestrates the pipeline (vision, config, LLM, execution).
    from session_handler import main as session_main
    session_main()

   


if __name__ == "__main__":
    main()