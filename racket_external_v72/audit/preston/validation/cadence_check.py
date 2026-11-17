"""Look for edited holds during first-pass accepted fit windows in the public playback."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('video', type=Path)
args = p.parse_args()
root = Path(__file__).resolve().parent
video = cv2.VideoCapture(str(args.video))
with (root.parent / 'anti_eligible.csv').open() as f, (root / 'cadence.csv').open('w') as out:
    writer = csv.writer(out)
    writer.writerow(['image_ref', 'phase','intervals','p05_changed_pixel_count','median_changed_pixel_count',
                     'near_static_pairs','nonzero_pairs'])
    for row in csv.DictReader(f):
        for short, phase in [('in','before'), ('out','after')]:
            start = int(row['win_start_' + short])
            end = int(row['win_end_' + short])
            video.set(cv2.CAP_PROP_POS_FRAMES,start)
            counts = []
            prev = None
            for idx in range(start,end+1):
                ok, frame = video.read()
                assert ok
                gray = cv2.cvtColor(frame[20:226,34:317],cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    diff = cv2.absdiff(gray,prev)
                    counts.append(int((diff > 16).sum()))
                prev = gray
            writer.writerow([row['id'],phase,len(counts),round(np.quantile(counts,.05),1),
                             round(np.median(counts),1),sum(i <= 10 for i in counts),
                             sum(i > 10 for i in counts)])
