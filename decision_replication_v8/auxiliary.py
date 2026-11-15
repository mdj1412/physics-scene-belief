"""Bounded, disjoint high-index acquisition assistance before allocation expiry."""
import sys,time
from pipeline import acquire,ROOT
start,stop=map(int,sys.argv[1:3]);assert (start,stop) in [(56,60),(60,64)]
deadline=time.monotonic()+240
for c in range(start,stop):
 if time.monotonic()>deadline-90:break
 if not (ROOT/'acquisition'/f'case_{c:03d}'/'complete.json').exists():acquire(c)
print('AUXILIARY_COMPLETE',flush=True)
