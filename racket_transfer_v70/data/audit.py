"""Audit original vs fixed Gossard split, experimental identifiers and zero-shot folds."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
CONFIG = {
    1: ("Hexer Duro", "inverted_offensive", "offensive", 2.1),
    2: ("Power 3", "inverted_allround", "offensive", 1.8),
    3: ("Hexer Duro", "inverted_offensive", "defensive", 2.1),
    4: ("Power 3", "inverted_allround", "defensive", 1.8),
    5: ("Grass D.TecS", "long_pips", "allround", 1.2),
    6: ("Grass D.TecS", "long_pips", "allround", 0.0),
    7: ("Diamant", "medium_pips", "allround", 1.2),
    8: ("Blowfish", "short_pips", "allround", 2.0),
    9: ("Hexer Duro", "inverted_offensive", "allround", 2.1),
    10: ("A-B-S II soft", "anti_spin", "allround", 2.1),
}
def read(split, rid):
    path = UPSTREAM / split / f"racket_{rid:02}.csv"
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [tuple(map(float, row)) for row in reader]
    return header, rows, hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    groups = {}
    hashes = {}
    all_rows = []
    key_to_rackets = defaultdict(set)
    split_map = []
    for rid, (rubber, kind, blade, sponge) in CONFIG.items():
        parts = {}
        for part in ("original", "train", "test"):
            header, rows, digest = read(part, rid)
            parts[part] = rows
            hashes[f"{part}/racket_{rid:02}.csv"] = digest
        orig, tr, te = (parts[part] for part in ("original", "train", "test"))
        total = set(orig)
        raw_train, raw_test = set(tr), set(te)
        assert len(total) == len(orig) and len(raw_train) == len(tr) and len(raw_test) == len(te)
        assert raw_train.isdisjoint(raw_test) and raw_train.union(raw_test) <= total
        assert all(all(math.isfinite(i) for i in row) for row in orig)
        assert len({row[:2] for row in orig}) == len(orig)
        for idx, row in enumerate(orig):
            part = "train" if row in raw_train else ("test" if row in raw_test else "original_only")
            split_map.append((rid, idx, int(row[0]), int(row[1]), part))
            key_to_rackets[row[:2]].add(rid)
            all_rows.append((rubber, row))
        groups[rid] = dict(rubber=rubber, surface=kind, blade=blade, sponge_mm=sponge,
            rows={p:len(v) for p,v in parts.items()},original_only=len(total-raw_train-raw_test))
    pre = [row[2:5]+row[8:11] for _, row in all_rows]
    rows = [r for _,r in all_rows]
    spans = {name: [min(row[ix] for row in rows),max(row[ix] for row in rows)] for ix,name in enumerate(header)}
    coverage = {}
    for rubber in sorted(set(rubber for rubber,_ in all_rows)):
        def incoming(row):
            return row[8], row[10], row[3]
        source = [incoming(row) for name,row in all_rows if name != rubber]
        target = [incoming(row) for name,row in all_rows if name == rubber]
        bounds = [(min(v), max(v)) for v in zip(*source)]
        outside = [any(x < lo or x > hi for x,(lo,hi) in zip(values,bounds)) for values in target]
        coverage[rubber] = {"impacts":len(target),"outside_3d_source_bounding_box":sum(outside)}
    stats = {"revision":"159c6f040dd7832570b537b3b5fd00a97fc28c15", "columns":header,
             "file_sha256":hashes,"configs":groups,"column_ranges":spans,
             "angle_counts":dict(Counter(int(row[0]) for row in rows)),
             "duplicate_incoming_6d_exact":len(pre)-len(set(pre)),
             "duplicate_12d_kinematics_exact":len(rows)-len(set(row[2:] for row in rows)),
             "duplicate_14column_numeric_rows":len(rows)-len(set(rows)),
             "racket_angle_traj_dir_keys":len(key_to_rackets),
             "such_keys_shared_by_multiple_configurations":sum(len(v)>1 for v in key_to_rackets.values()),
             "source_bounding_box_in_vx_vz_omega_y":coverage}
    (ROOT/"audit.json").write_text(json.dumps(stats,indent=2)+"\n")
    with (ROOT/"official_membership.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["configuration","original_row_0based","racket_angle","traj_dir_as_int","official_membership"])
        writer.writerows(split_map)
if __name__=="__main__":
    main()
