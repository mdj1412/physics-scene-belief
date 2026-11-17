"""Reconstruct supervised shot rows from the official 640x480 public movie."""

import argparse
import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def track(ref):
    from track_centers import track as analyze

    analyze(
        int(ref["start_frame"]),
        int(ref["end_frame_exclusive"]),
        f"r{int(ref['image_ref']):02d}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.environ["PRESTON_VIDEO"] = str(args.video.resolve())
    os.environ["PRESTON_OUTPUT"] = str(args.output.resolve())
    from paths import CACHE
    import fit_kinematics
    import assess_sensitivity

    CACHE.mkdir(parents=True, exist_ok=True)
    with Path(__file__).with_name("reference_frames.csv").open() as f:
        refs = list(csv.DictReader(f))
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(track, refs))
    fit_kinematics.run()
    assess_sensitivity.run()
