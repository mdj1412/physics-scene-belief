"""Re-localize static, masked surface patches under calibrated real directional light."""
from pathlib import Path
import csv, hashlib, io, json, os, sys, zipfile
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import minimum_filter
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=Path('/scratch2/mdj1412/datasets/diligent_v64')
sys.path.insert(0,str(next(DATA.glob('pypng*.whl'))))
import png

def read_image(z, name, intensity):
    width,height,rows,meta=png.Reader(bytes=z.read(name)).asDirect()
    assert meta['bitdepth']==16 and meta['planes']==3
    rgb=np.vstack([np.frombuffer(r,dtype=np.uint16) for r in rows]).reshape(height,width,3)
    return (rgb/intensity).mean(-1)

def compare(image, reference, x, y):
    patch=reference[y-7:y+8,x-7:x+8]
    tile=image[y-15:y+16,x-15:x+16]
    windows=sliding_window_view(tile,(15,15))
    zz=patch-patch.mean()
    centered=windows-windows.mean(axis=(-1,-2),keepdims=True)
    den=np.sqrt((centered**2).sum((-1,-2))*(zz**2).sum())
    ncc=np.divide(np.einsum('xyij,ij->xy',centered,zz),den,
                  out=np.full_like(den,-2.),where=den>0)
    if ncc.max() < -1: return None
    # Exact or numerical ties use the spatially closest match.
    iy,ix=np.where(ncc>=ncc.max()-1e-12)
    j=np.argmin((ix-8)**2+(iy-8)**2)
    return [int(ix[j]-8),int(iy[j]-8),float(ncc[iy[j],ix[j]])]

def main():
    assert os.getenv("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    target_angles=np.deg2rad([-40,-15,15,40])
    target=np.array([np.sin(target_angles),np.zeros(4),np.cos(target_angles)]).T
    results=[]; example=None
    archive=DATA/'DiLiGenT.zip'
    outrows=[]; selections={}
    with zipfile.ZipFile(archive) as z:
        paths=sorted(p for p in z.namelist() if '/pmsData/' in p and p.endswith('/light_directions.txt') and '__MACOSX' not in p)
        assert len(paths)==10,paths
        for p in paths:
            folder=p.rsplit('/',1)[0]+'/'
            name=folder.split('/')[-2]
            files=z.read(folder+'filenames.txt').decode().split()
            light=np.loadtxt(io.BytesIO(z.read(p)))
            intensity=np.loadtxt(io.BytesIO(z.read(folder+'light_intensities.txt')))
            mask=np.asarray(Image.open(io.BytesIO(z.read(folder+'mask.png'))))
            if mask.ndim==3:mask=mask[:,:,0]
            interior=minimum_filter(mask>0,size=31,mode='constant',cval=False)
            assert light.shape==intensity.shape==(96,3) and len(files)==96
            unit=light/np.linalg.norm(light,axis=1)[:,None]
            ix=np.argmax(target@unit.T,axis=1)
            assert len(set(ix))==4
            pics=[read_image(z,folder+files[i],intensity[i]) for i in ix]
            selections[name]=dict(filenames=[files[i] for i in ix],
                unit_directions=unit[ix].tolist(),
                angle_error_deg=np.rad2deg(np.arccos(np.clip((target*unit[ix]).sum(1),-1,1))).tolist())
            for rev in (False,True):
                imgs=pics[::-1] if rev else pics
                offsets=[]; self_errors=0
                for y in range(15,mask.shape[0]-15,25):
                    for x in range(15,mask.shape[1]-15,25):
                        if not interior[y,x]:continue
                        matches=[compare(img,imgs[0],x,y) for img in imgs]
                        if any(m is None for m in matches):continue
                        self_errors+=int(matches[0][0]!=0 or matches[0][1]!=0)
                        offsets.append(matches[1:])
                        for k,(dx,dy,ncc) in enumerate(matches):
                            outrows.append([name,int(rev),x,y,k,dx,dy,ncc])
                a=np.array(offsets)
                assert len(a)>0 and self_errors==0,(name,self_errors)
                for threshold in (.7,.8,.9):
                    accepted=(a[:,:,2]>=threshold).all(1)
                    shifted=(a[:,:,0]>=2).all(1)|(a[:,:,0]<=-2).all(1)
                    results.append(dict(object=name,reverse=rev,threshold=threshold,
                         valid=len(a),accepted=int(accepted.sum()),consistent_shift=int((accepted&shifted).sum()),
                         same_image_nonzero=self_errors))
    with (ROOT/'static_tracks64.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['object','reverse','x','y','light_index','dx','dy','NCC']);w.writerows(outrows)
    checksum=hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024),b''):
            checksum.update(chunk)
    digest=checksum.hexdigest()
    result=dict(archive_sha256=digest,objects=len(paths),directions=selections,per_object=results)
    (ROOT/'static_tracks64.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
