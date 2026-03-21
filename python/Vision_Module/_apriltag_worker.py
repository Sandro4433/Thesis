# _apriltag_worker.py
# ─────────────────────────────────────────────────────────────────────────────
# Standalone worker: receives a grayscale image array via a pickle file,
# runs pupil_apriltags detection, and writes the results to a second pickle
# file.  Runs in its own process so a segfault from libapriltag does not
# kill the main Vision_Main process.
#
# Usage (called by Vision_Main — not intended for direct use):
#   python _apriltag_worker.py <input_pkl> <output_pkl>
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import pickle
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: _apriltag_worker.py <input_pkl> <output_pkl>", file=sys.stderr)
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]

    # Load the grayscale array
    with open(in_path, "rb") as f:
        gray = pickle.load(f)

    # Run detection
    from pupil_apriltags import Detector  # type: ignore
    detector = Detector(
        families="tag25h9",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )
    detections = detector.detect(gray)
    del detector

    # Write results — pickle the detection objects directly
    with open(out_path, "wb") as f:
        pickle.dump(detections, f)


if __name__ == "__main__":
    main()