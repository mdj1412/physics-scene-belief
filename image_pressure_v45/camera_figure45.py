"""Show literal unannotated pre-impact frames & corresponding point estimates."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
inputs=root/'camera45'/'panel_00'
fig,axes=plt.subplots(2,3,figsize=(9,3.4),layout="constrained")
for i,d in enumerate((0,2)):
    frames=np.load(inputs/f'pixels_fold_hollow_h31_d{d}.npz')['frames']
    a=np.load(inputs/'fold_hollow_h31.npz')
    x=a['x_hat_mm'][d,0,0]; y=a['oracle_x_mm'][d,0,0]
    for j in range(3):
        ax=axes[i,j];ax.imshow(frames[j],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        if i==0:ax.set_title(f'Pre-impact frame {j+1}',fontsize=11)
        if j==0:ax.set_ylabel(f'True {y:.2f} mm\nVisual {x:.2f} mm',fontsize=9)
fig.savefig(root/'camera_examples45.png',dpi=165)
