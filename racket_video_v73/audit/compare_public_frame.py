"""Compare a public thumbnail to continuous earlier public video. Usage: python FILE_MOVIE FILE_IMAGE."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np

def compare(movie, thumbnail):
    width, height = 160, 120
    target = cv2.imread(str(thumbnail), cv2.IMREAD_GRAYSCALE)
    target = cv2.resize(target, (width, height), interpolation=cv2.INTER_AREA).astype(float)
    with subprocess.Popen(['ffmpeg','-v','error','-i',str(movie),'-vf',f'scale={width}:{height}:flags=area,format=gray','-f','rawvideo','-'],stdout=subprocess.PIPE) as proc:
        i = 0
        best = []
        while True:
            chunk=proc.stdout.read(width*height)
            if not chunk:break
            img=np.frombuffer(chunk,dtype=np.uint8).reshape((height,width))
            err=float(np.mean(np.abs(img-target)))
            best.append((err,i))
            i+=1
        assert proc.wait()==0
    return {'movie_sha256':hashlib.sha256(movie.read_bytes()).hexdigest(),
            'thumbnail_sha256':hashlib.sha256(thumbnail.read_bytes()).hexdigest(),
            'decoded_frames':i,'metric':'mean absolute 8-bit grayscale difference; 160x120 full frame',
            'nearest':[{'frame_zero_based':i, 'mae':round(err,5)} for err,i in sorted(best)[:5]]}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('movie',type=Path);p.add_argument('thumbnail',type=Path)
    args=p.parse_args()
    print(json.dumps(compare(args.movie,args.thumbnail),indent=2))
