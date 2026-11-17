"""Render signed-offset contrasts, intervals and primary compute results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from stats42 import ROOT, OFF, NZ, FAMILY, HEIGHT

def read(name):return json.loads((ROOT/name).read_text())
def band(x,idx=None,nd=6):
    vals=[np.asarray(x[k]) for k in ('mean','lo','hi')]
    if idx is not None: vals=[v[idx] for v in vals]
    m,l,h=vals
    return f'{m:+.{nd}f} [{l:+.{nd}f}, {h:+.{nd}f}]'

def fig(s):
    fig,axs=plt.subplots(1,3,figsize=(15.5,4.8),gridspec_kw=dict(width_ratios=[1.08,1.15,1]))
    colors=('#176383','#ad4d28','#327c5e','#755698')
    for i,f in enumerate(FAMILY):
        for j,h in enumerate(HEIGHT):
            color=colors[2*i+j]
            label=f'{f}, {h[1:]} mm'
            axs[0].plot(OFF,100*np.asarray(s['group_flip']['mean'])[i,j],color=color,
                        marker=('o','s')[j],ms=3.5,lw=1.4,label=label)
            val,lo,hi=[np.asarray(s['group_geometry'][k])[i,j]*1000 for k in ('mean','lo','hi')]
            axs[1].plot(OFF,val,lw=1.5,color=color,alpha=.8,label=label)
            for x,y,l,u in zip(OFF,val,lo,hi):
                axs[1].plot(x,y,marker=('o','s')[j],ms=4.8,color=color,mfc=(color if l*u>0 else 'white'))
    axs[0].set(ylabel='Contour vs base event flips (%)',title='A. When contour changes the event')
    axs[0].legend(frameon=False,fontsize=8)
    axs[1].set(ylabel='Projected − original Brier (×1,000)',title='B. When projection increases loss')
    axs[1].axhline(0,color='0.34',ls='--',lw=1)
    val,lo,hi=[np.array(s['by_offset_selector'][k])*1000 for k in ('mean','lo','hi')]
    axs[2].errorbar(OFF,val,yerr=[val-lo,hi-val],fmt='o-',capsize=3,lw=1.4,ms=4,color='#195a75')
    axs[2].axhline(0,color='0.34',ls='--',lw=1)
    axs[2].set(ylabel='Selector − always zero Brier (×1,000)',title='C. Three-way correction selection')
    for ax in axs:
        ax.set_xlabel('Actual − nominal contact height (mm)')
        ax.spines[['right','top']].set_visible(False)
        ax.set_xticks([ -1,-.5,0,.5,1])
        ax.grid(axis='y',alpha=.17)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(ROOT/'offset42.png',dpi=170,bbox_inches='tight');plt.close(fig)

def main():
    s=read('stats42.json');reveal=read('reveal42.json');soft=read('soft42.json')
    fig(s)
    rows=['## Signed placement error와 물리적 전환','',
        '아래 flip은 같은 물체·같은 signed offset에서 hollow 또는 bulge가 base와 다른 실제 event를 낸 비율이다. 무변형 base는 flip 비율의 분모에서 제외했다. Brier는 base까지 포함한 24개 cell의 동일 가중 평균이다. 양의 Brier 차이는 앞의 방법의 손해다. 각 Δ 열의 구간은 9개 signed offset에 대한 동시 95% 구간이다.','',
        '| δ (mm) | Contour event flip | 자기 contour의 δ=0 대비 flip | Original 0% | Estimated 0% | Estimated selector | Estimated−original, 0% | Selector−0%, estimated |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    risk=np.array(s['mean_risk_by_offset'])
    for k,x in enumerate(OFF):
        rows.append(f"| {x:+g} | {s['flip']['mean'][k]:.1%} | {s['self_flip']['mean'][k]:.1%} | {risk[k,0,0]:.5f} | {risk[k,2,0]:.5f} | {risk[k,2,3]:.5f} | {band(s['geometry'],k,5)} | {band(s['by_offset_selector'],k,5)} |")
    rows+=['','### 어느 물체에서 projection이 손해였는가','',
           '같은 0% 보정에서 estimated−original의 Brier 차이(×1,000)다. **굵은 값**은 36개 사전 대조를 동시에 보정한 구간에서 부호가 결정된 셀이다. 높이는 flare·roll의 명목 접촉 위치이고 첫 행은 실제 이동량이다.','',
           '| Shape·높이 / δ (mm) | '+' | '.join(f'{x:+g}' for x in OFF)+' |','|---|'+'---:|'*len(OFF)]
    for i,f in enumerate(FAMILY):
        for j,h in enumerate(HEIGHT):
            cells=[]
            for k in range(9):
                m,l,u=[np.array(s['group_geometry'][z])[i,j,k] for z in ('mean','lo','hi')]
                v=f'{1000*m:+.2f}';cells.append(f'**{v}**' if l*u>0 else v)
            rows.append(f"| {f} / {h[1:]} mm | "+' | '.join(cells)+' |')
    rows+=['','### 보정을 고르는 이득과 계산 비용','',
           '아래 Brier는 명목 높이를 제외한 8개 signed offset을 동일하게 가중한 주 분석이다. Oracle은 reference profile의 실제 변형 방향을 알려 준 decoder다.','',
           '| Decoder | 항상 0% | 항상 32.5% | 항상 v38 optimizer | 세 정책 selector | Selector−0%, paired 95% |','|---|---:|---:|---:|---:|---:|']
    for d,name in enumerate(('Original','Oracle projection','Estimated projection')):
        rows.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in s['mean_risk_nonzero'][d])+' | '+band(s['main_by_decoder'],d)+' |')
    rows+=['',f"Primary selector−always-zero는 {band(s['main'])}였다. 12개 panel 중 {s['wins']}개에서 selector의 nonzero-offset 평균 Brier가 더 낮았다. 9개 높이를 모두 포함하면 {band(s['all9'])}다. Estimated 안에서 selector−always-fixed와 selector−always-optimizer의 두 동시 95% 구간은 각각 {band(s['selected_minus_fixed_opt'],0)} 및 {band(s['selected_minus_fixed_opt'],1)}였다.",'',
           '| 동일 GPU, 1,024개 물체 | Warm p90, ms [95%] | Cold p90, ms |','|---|---:|---:|']
    w=s['timing']['full_ms'];cold=s['timing']['cold_ms']
    for k,name in enumerate(('Original + 0%','Estimated + 0%','Estimated + 32.5%','Estimated + optimizer','Estimated + selector')):
        l,h=w['ci'][k]
        rows.append(f"| {name} | {w['p90'][k]:,.2f} [{l:,.2f}, {h:,.2f}] | {cold['p90'][k]:,.2f} |")
    lo,hi=w['ratio_ci']
    rows+=['',f"Warm selector / zero p90 ratio는 {w['ratio']:.4f}× [{lo:.4f}, {hi:.4f}]였다. 한 GPU에서 12개 panel × 4회 순서를 섞어 측정했고 panel을 함께 bootstrap했다. Warm에는 acquisition, posterior inference, 실제 correction 결정, 장치 동기화가 들어간다. Cold는 메모리에 있는 32쌍의 direction fitting과 template setup까지 더한 값이다. 물리적 촬영·정렬과 최초 bank 생성은 두 값에서 제외했다. [전체 수치](stats42.json)",'',
        '### 실제 높이를 알려 주거나 변형 방향을 부분적으로 보존하면','',
        '첫 32개 물체와 첫 noise만 미리 정해 12개 panel × 24개 cell에서 보조 비교했다. 숨은 높이 control은 GPU와 같은 확률을 CPU로 재구성했다. Revealed control에는 실제 높이를 주고 동일 posterior의 source hypothesis마다 stick event만 새로 계산했다. Finite-variance는 같은 profile pair의 분산에서 sensor noise를 빼고, 해당 축 정보를 일부 남기는 v41 likelihood다. 모두 0% 보정이다.','',
        '| δ (mm) | 숨긴 높이 original | 숨긴 높이 estimated | 실제 높이 original | 실제 높이 estimated | 숨긴 높이 finite-variance |','|---:|---:|---:|---:|---:|---:|']
    rm=np.array(reveal['mean']);sm=np.array(soft['risk'])
    np.testing.assert_allclose(rm[0],sm[:,:3],atol=4e-12,rtol=0)
    for i,x in enumerate(OFF):
        rows.append(f'| {x:+g} | '+' | '.join(f'{z:.5f}' for z in (rm[0,i,0],rm[0,i,2],rm[1,i,0],rm[1,i,2],sm[i,3]))+' |')
    rows+=['',f"명목 높이에서 CPU와 GPU 확률의 최대 차이는 {reveal['max_nominal_reproduction_error']:.2e}였다. 이 대조는 실제 높이만 알려 주며 source bank가 가진 quadratic contour 및 friction 가정은 유지한다. 유효한 접촉 기하를 완전히 아는 oracle로 해석할 수 없다. Finite-variance는 주 실험 실행 중 명세한 보조 baseline이다. [Readout control](reveal42.json) · [Finite-variance 결과](soft42.json)"]
    (ROOT/'TABLES_KO.md').write_text('\n'.join(rows)+'\n')
if __name__=='__main__':main()
