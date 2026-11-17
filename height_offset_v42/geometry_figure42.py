"""A geometry-only illustration, selected before reading any target loss."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from slope_forecast42 import displacement

ROOT=Path(__file__).resolve().parent

def main():
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),layout='constrained')
    true='#205c78'; fit='#b64c22'
    for i,(h,ch) in enumerate(((19,'19 mm'),(31,'31 mm'))):
        # The seven points (at nominal height) stay fixed for every hidden offset.
        dy=np.arange(-6.,7.,2.)
        samples=np.array([displacement('flare',h+d)[0] for d in dy])
        coefs=np.polyfit(dy,samples,2)
        ax=axes[i,0]
        x=np.linspace(-6.25,6.25,401)
        ax.axvspan(-1,1,color='#dfdcce',alpha=.55,lw=0,label='Hidden contact interval')
        ax.axvline(0,color='.6',lw=1,ls=':')
        ax.plot(x,[displacement('flare',h+d)[0] for d in x],color=true,label='Continuous deformation',lw=2)
        ax.plot(x,np.polyval(coefs,x),color=fit,label='Quadratic fit to 7 points',lw=1.6,ls='--')
        ax.scatter(dy,samples,color=true,edgecolors='white',lw=.5,s=29,zorder=4,label='Registered samples')
        ax.set(title=f'Flare / {ch}: registered contour',ylabel='Bulge − base offset (mm)',xlabel='Position from nominal center (mm)',
               xticks=[-6,-4,-2,0,2,4,6])
        if i==0:ax.legend(loc='upper right',fontsize=7.5,framealpha=.95)
        ax=axes[i,1]
        x=np.linspace(-1,1,401)
        mm=np.array([-1,-.5,-.25,-.125,0,.125,.25,.5,1])
        ax.axhline(0,color='.55',lw=1)
        ax.axvline(0,color='.6',lw=1,ls=':')
        ax.axhline(coefs[1],color=fit,ls='--',label='Apparent nominal slope',lw=1.6)
        ax.plot(x,[displacement('flare',h+d)[1] for d in x],color=true,lw=2,label='Actual local slope')
        ax.scatter(mm,[displacement('flare',h+d)[1] for d in mm],color=true,s=27,edgecolors='white',lw=.5,zorder=4)
        ax.set(title=f'Flare / {ch}: slope at contact',ylabel='Bulge − base slope (mm/mm)',xlabel='Actual − nominal height (mm)',
               xticks=[-1,-.5,0,.5,1])
        if i==0:ax.legend(loc='lower left',fontsize=8,framealpha=.95)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    fig.savefig(ROOT/'geometry42.png',dpi=175)
    plt.close(fig)
if __name__=='__main__':main()
