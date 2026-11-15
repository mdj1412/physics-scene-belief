"""Descriptive profile displacement, holding the common latent base contour fixed."""
from pathlib import Path
import json
import numpy as np


def deterministic_part(u):
    return {
        "shelf": .043*.025*(u-2*np.sin(np.pi*u/2)/np.pi),
        "shoulder": .072*.025*(u-np.arctan(u)),
        "ridge": -.153*.025*.5*u*u/(1+u*u),
        "bead": -.078*.025*.8**2*(1-np.exp(-.5*(u/.8)**2)),
    }


def main():
    out = {}
    for mode,lo,hi,n,noise in (("factual",-.010,.010,9,.0016),("query_view",.019,.031,7,.00030)):
        values=deterministic_part(np.linspace(lo,hi,n)/.025)
        out[mode] = {}
        for a,b in (("shelf","shoulder"),("ridge","bead"),("shelf","ridge")):
            d=values[a]-values[b]
            rms=np.sqrt(np.mean(d*d))
            out[mode][a+"-"+b]=dict(rms_m=float(rms),max_abs_m=float(abs(d).max()),
                                    sigma_m=noise,rms_over_sigma=float(rms/noise))
    (Path(__file__).resolve().parent/"geometry_shift39.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
