"""Recompute manually audited shot centres independently of the angular-correlation tracker."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("video", type=Path)
v = cv2.VideoCapture(str(p.parse_args().video))
root = Path(__file__).resolve().parent
clips = {3: (175.05, 179.9), 15: (349.05, 352.3), 17: (374.65, 377.9)}
for ref, (start, stop) in clips.items():
    v.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = []
    for i in range(round((stop - start) * 30)):
        ok, frame = v.read()
        assert ok
        frames.append(cv2.cvtColor(frame[20:226, 34:317], cv2.COLOR_BGR2GRAY))
    background = np.median(frames, axis=0).astype('uint8')
    centres = {}
    for i, frame in enumerate(frames):
        mask = (cv2.absdiff(frame, background) > 22).astype('uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if 14 <= w <= 35 and 14 <= h <= 35 and area > 90:
                candidates.append((area, x + w / 2, y + h / 2, w, h))
        if candidates:
            centres[str(i)] = list(max(candidates)[1:3])
    (root / f'centres_ref{ref}.json').write_text(json.dumps(centres, indent=2) + '\n')
