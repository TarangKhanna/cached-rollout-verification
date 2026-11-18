#!/usr/bin/env python3
"""Reproducible figures for the narrowly scoped SPEC-RL manuscript.
Tabular curves are algebraic identities. Finite points are exact independent
enumerations, apart from the appendix batch study (paired Monte Carlo).
"""
from pathlib import Path
import numpy as np
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG=Path(__file__).resolve().parent.parent/'figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight',
 'font.family':'DejaVu Sans','pdf.fonttype':42})

th=np.linspace(0,2.0,500)
t=np.exp(-th/2)
b=(3-2*t-t*t)/8
bf=(1-t)/6
fig,ax=plt.subplots(figsize=(6.0,2.65))
ax.plot(th,b,lw=2,label='Unfiltered reward-observable discrepancy')
ax.plot(th,bf,lw=2,ls='--',label='Actor-matched mixed-filter discrepancy')
ax.axvline(np.log(4),lw=1,ls=':',color='gray')
ax.scatter([np.log(4),np.log(4)],[7/32,1/12],s=35,zorder=4)
ax.annotate(r'$\theta=\log 4$',xy=(np.log(4),.19),xytext=(-38,17),
 xycoords='data',textcoords='offset points',fontsize=8,
 arrowprops={'arrowstyle':'-','lw':.6})
ax.set(xlabel=r'Effective logit step $\theta$',ylabel='Difference in mean binary reward',
 xlim=(0,2),ylim=(0,.31))
ax.grid(alpha=.14)
ax.legend(loc='upper left',frameon=False,fontsize=8)
fig.savefig(FIG/'tabular_limit.pdf'); fig.savefig(FIG/'tabular_limit.png',dpi=180); plt.close(fig)

finite=json.loads((FIG.parent/'data'/'finite_exact_values.json').read_text())
eta=np.array([r['eta'] for r in finite])
reward_pp=100*np.array([r['fresh_second_minus_included_second'] for r in finite])
filtered_pp=100*np.array([r['matched_filter_two_step_fresh_advantage'] for r in finite])
fresh_yield=np.array([r['fresh_group_yield'] for r in finite])
include_yield=np.array([r['included_group_yield'] for r in finite])
fig,axs=plt.subplots(1,2,figsize=(6.8,2.8))
axs[0].plot(eta,reward_pp,'o-',label='Unfiltered next-update gap')
axs[0].plot(eta,filtered_pp,'s--',label='Matched mixed-filter gap')
axs[0].set(xlabel='First update size $\\eta$',ylabel='Fresh minus reuse (percentage points)',xlim=(-.015,.84))
axs[0].legend(fontsize=7,frameon=False)
axs[1].plot(eta,fresh_yield,'o-',label='Fresh')
axs[1].plot(eta,include_yield,'s--',label='Training-used cache')
axs[1].set(xlabel='First update size $\\eta$',ylabel='Probability of mixed-reward group',ylim=(.45,.80),xlim=(-.015,.84))
axs[1].legend(fontsize=7,frameon=False)
for ax in axs:ax.grid(alpha=.14)
fig.tight_layout(w_pad=2)
fig.savefig(FIG/'exact_two_update.pdf');fig.savefig(FIG/'exact_two_update.png',dpi=180);plt.close(fig)

# Read the reproduced seeded Monte Carlo log, never synthesize confidence bars.
batch_log=(FIG.parent/'data'/'exploratory'/'batch_sweep.txt').read_text()
rows=[]
for line in batch_log.splitlines():
    if line.startswith('eta=0.05'):
        m=re.search(r'B=\s*(\d+) included_minus_fresh=([+-]?[0-9.]+) \+/- ([0-9.]+)',line)
        if m: rows.append((int(m.group(1)),abs(float(m.group(2))),float(m.group(3))))
assert len(rows)==7, f'Expected seven eta=0.05 batch rows, found {len(rows)}'
B=np.array([r[0] for r in rows])
delta=np.array([r[1] for r in rows])
err=np.array([r[2] for r in rows])
fig,ax=plt.subplots(figsize=(5.2,2.7))
ax.errorbar(B,delta,yerr=err,fmt='o-',capsize=2,label='Paired MC, 300 actor updates per batch size')
ax.plot(B,delta[0]/B,ls='--',label=r'$1/B$ reference (anchored at $B=1$)')
ax.set_xscale('log',base=2);ax.set_yscale('log')
ax.set(xlabel='Number of independent old-policy groups in first update $B$',
 ylabel='Absolute mean projected gradient gap')
ax.grid(alpha=.2,which='both');ax.legend(fontsize=7,frameon=False)
fig.tight_layout()
fig.savefig(FIG/'batch_dilution.pdf');fig.savefig(FIG/'batch_dilution.png',dpi=180);plt.close(fig)
print('generated',sorted(str(p.name) for p in FIG.glob('*.pdf')))
