#!/usr/bin/env python3
from itertools import groupby

import numpy as np
import argparse
import os


def get_frames(out_file):
    """Parse an output file into a list of frames, each a list of (x, y) tuples."""
    num_parts, box_size = out_file.readline().split()
    num_parts = int(num_parts)
    box_size = float(box_size)

    frames = []
    for is_data, lines in groupby(out_file, key=lambda line: bool(line.strip())):
        if not is_data:
            continue
        frame = []
        for line in lines:
            x, y = line.split()
            frame.append((float(x), float(y)))
        frames.append(frame)

    for frame in frames:
        assert len(frame) == num_parts

    return frames


def calculate_dist(frames1, frames2):
    """Return the mean particle-wise Euclidean distance for each pair of frames."""
    assert len(frames1) == len(frames2)

    avg_dists = []
    for f1, f2 in zip(frames1, frames2):
        a1 = np.array(f1)
        a2 = np.array(f2)
        dists = np.sqrt((a1[:, 0] - a2[:, 0]) ** 2 + (a1[:, 1] - a2[:, 1]) ** 2)
        avg_dists.append(np.mean(dists))

    return avg_dists


def check_conditions(avg_dists):
    """Check that the test output agrees with the reference output.

    1. The mean of the per-frame average distances over the first 50 frames
       must be below 3e-7 (early frames must be almost identical).
    2. The mean of the per-frame average distances over all frames must be
       below 0.025 (the overall deviation must stay small).
    """
    assert np.mean(avg_dists[:50]) < 3e-7
    assert np.mean(avg_dists) < 0.025


def main():
    parser = argparse.ArgumentParser(description="Check the correctness of HW2 output.")
    parser.add_argument("test_out", type=argparse.FileType(), help="Your code's output")
    parser.add_argument("verf_out", type=argparse.FileType(), help="O(n^2) code's output")
    args = parser.parse_args()

    test_frames = get_frames(args.test_out)
    verf_frames = get_frames(args.verf_out)

    avg_dists = calculate_dist(test_frames, verf_frames)
    check_conditions(avg_dists)

    print("Check between {} and {} succeeded!".format(
        os.path.basename(args.test_out.name), os.path.basename(args.verf_out.name)))


if __name__ == "__main__":
    main()
