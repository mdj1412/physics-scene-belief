"""Audit Rod Cross's publicly linked 600 fps shot. Requires Python 3 and ffmpeg."""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

WIDTH, HEIGHT = 640, 480
BALL_DIAMETER = 0.060
CAPTURE_RATE = 600


def read_pixels(path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-pix_fmt", "rgb24",
           "-f", "rawvideo", "-"]
    return subprocess.check_output(cmd)


def ball_box(frame, red_cut, ref_frames):
    # The wooden support below row 310 shares the orange hue of the ball.
    # A black background, a dark board, and the ball occupy this fixed ROI.
    xcounts, ycounts = [0] * WIDTH, [0] * HEIGHT
    for y in range(105, 310):
        for x in range(WIDTH):
            i = 3 * (WIDTH * y + x)
            r, g, b = frame[i:i + 3]
            if (r > red_cut and r - g > 38 and r - b > 50
                    and r - min(ref_frames[0][i], ref_frames[1][i]) > 45):
                xcounts[x] += 1
                ycounts[y] += 1
    xs = [i for i, count in enumerate(xcounts) if count >= 7]
    total = sum(ycounts)
    running = 0
    for centre_y, count in enumerate(ycounts):
        running += count
        if running * 2 >= total:
            break
    ys = [i for i, count in enumerate(ycounts) if count >= 7 and abs(i - centre_y) <= 35]
    return min(xs), max(xs), min(ys), max(ys)


def slope_per_frame(values):
    t_mean = (len(values) - 1) / 2
    return (sum((i - t_mean) * v for i, v in enumerate(values))
            / sum((i - t_mean) ** 2 for i in range(len(values))))


def measure(boxes, indices, diameter_px):
    x = [(boxes[i][0] + boxes[i][1]) / 2 for i in indices]
    y = [(boxes[i][2] + boxes[i][3]) / 2 for i in indices]
    factor = BALL_DIAMETER * CAPTURE_RATE / diameter_px
    vx = slope_per_frame(x) * factor
    vy = -slope_per_frame(y) * factor
    return {"vx_mps": vx, "vy_mps": vy,
            "speed_mps": math.hypot(vx, vy), "angle_deg": math.degrees(math.atan2(vy, vx))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("movie", type=Path)
    args = parser.parse_args()
    data = read_pixels(args.movie)
    stride = WIDTH * HEIGHT * 3
    assert len(data) == 22 * stride, "Unexpected resolution or source movie"
    sensitivities = []
    ref_frames = (memoryview(data)[:stride], memoryview(data)[-stride:])
    for threshold in (60, 85, 110, 135):
        boxes = [ball_box(memoryview(data)[i:i + stride], threshold, ref_frames)
                 for i in range(0, len(data), stride)]
        # Full ball visible at frames 1-3. At 4 it approaches the board.
        # Frames 10-21 are visually clear of the board and the ball is fully in frame.
        free = [1, 2, 3] + list(range(10, 22))
        diameter = statistics.median([(boxes[i][1] - boxes[i][0] + 1 +
                                       boxes[i][3] - boxes[i][2] + 1) / 2 for i in free])
        conditions = {}
        for name, ids in [("pre_1_3", range(1, 4)), ("post_9_12", range(9, 13)),
                          ("post_10_14", range(10, 15)), ("post_10_21", range(10, 22))]:
            conditions[name] = measure(boxes, ids, diameter)
        sensitivities.append({"threshold": threshold, "diameter_px": diameter,
                              "windows": conditions,
                              "boxes_xyxy": [[a,c,b,d] for a,b,c,d in boxes]})
    print(json.dumps({"movie_sha256": hashlib.sha256(args.movie.read_bytes()).hexdigest(),
                      "capture_fps": CAPTURE_RATE, "ball_diameter_m": BALL_DIAMETER,
                      "analyses": sensitivities}, indent=2))


if __name__ == "__main__":
    main()
