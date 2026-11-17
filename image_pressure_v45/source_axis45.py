"""Calibrate one shape-blind whitened profile axis at each public query height."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "contact_patch_v44" / "results"

def main():
    artifacts = {}
    for height in (19,31):
        # Seven coordinates share one calibration standard deviation; its scale
        # cancels on normalization. Use one homologous surface/law per prototype.
        covariance = np.zeros((7,7))
        count = 0
        directions = []
        for panel in range(12):
            for family in ("fold","ripple"):
                stem = SOURCE / f"panel_{panel:02d}" / f"{family}_inlay_rational"
                with np.load(f"{stem}_hollow_h{height}.npz") as a, np.load(f"{stem}_bulge_h{height}.npz") as b:
                    np.testing.assert_array_equal(a["theta"],b["theta"])
                    delta = (b["query_paid"][:,:,0]-a["query_paid"][:,:,0]).reshape(-1,7)
                measured = delta[np.isfinite(delta).all(axis=1)]
                unit = measured / np.linalg.norm(measured,axis=1,keepdims=True)
                covariance += unit.T @ unit
                count += len(unit)
                directions.append(unit.mean(axis=0))
        values,vectors = np.linalg.eigh(covariance/count)
        unit = vectors[:,-1]
        if unit[np.argmax(abs(unit))]<0: unit=-unit
        cosines = np.abs(np.array(directions) @ unit).reshape(12,2)
        artifacts[str(height)] = dict(axis=unit.tolist(),variance_fraction=float(values[-1]/values.sum()),n=count,
                                      historical_abs_cosine_per_family=cosines.mean(axis=0).tolist())
    (ROOT / "source_axis45.json").write_text(json.dumps(artifacts,indent=2)+"\n")

if __name__ == "__main__":
    main()
