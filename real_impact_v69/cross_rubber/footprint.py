"""Bound the visible swept chalk footprint in the author's native crop."""
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

p = Path(sys.argv[1])
image = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(p),
                                 "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-"])
width, height = 410, 240
assert len(image) == width * height
# Median filter suppresses the board's printed texture.
rows = [[0] * width for _ in range(height)]
for y in range(20, 221):
    for x in range(15, 390):
        rows[y][x] = statistics.median(image[j * width + i] for j in range(y-2, y+3)
                                       for i in range(x-2, x+3))
segments = []
for threshold in (85, 95, 105, 115):
    widths = []
    starts = []
    ends = []
    ys = []
    for y in range(20, 221):
        xs = [x for x in range(15,390) if rows[y][x] < threshold]
        if len(xs) >= 12:
            # Bridge the narrow white reference line traversing the middle of the mark.
            starts.append(min(xs))
            ends.append(max(xs))
            widths.append(max(xs)-min(xs)+1)
            ys.append(y)
    w = max(ends) - min(starts) + 1
    h = max(ys) - min(ys) + 1
    segments.append({"gray_threshold":threshold, "length_px":w, "width_px":h,
                     "silhouette_area_px2":sum(widths),"aspect":w/h})
print(json.dumps({"image_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                  "segments":segments}, indent=2))
