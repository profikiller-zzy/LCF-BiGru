import sys, json, numpy as np
sys.path.insert(0,'tools/layer_screening'); sys.path.insert(0,'.')
from common import Panel, auroc
from stage2_matrices import build_bank
from stage2_length_controlled import load_controls
from stage5_nested_eval import load_flat
from dual_concept.backbone import StateCache
from dual_concept.concepts import concept_scores
from dual_concept.fusion import FusionConfig, build_feature_matrix, fit_sparse_fusion
from pathlib import Path
cache='/root/rivermind-data/dccf/cache_v2'
P=Panel('data/dccf_v1',cache); C=StateCache(Path(cache)/'rows')
wb_rows,wb_st=load_flat(Path('data/dccf_v1_wrapper_controls/development'),['wrapped_benign'],C,P.protocol)
tg={g for g,s,k in zip(P.group,P.split,P.kind) if s=='train' and k=='harmful'}
bank,_,_=build_bank(P,tg,[13,14,22])      # layers 14,15,23 for a quick look
fit=np.flatnonzero(P.split=='train'); wb_tr=[i for i,r in enumerate(wb_rows) if r['split']=='train']
L=[13,14,22]
r,m=concept_scores(P.states[fit][:,L],bank); rw,mw=concept_scores(wb_st[wb_tr][:,L],bank)
cols=[14,15,23]
print('per-layer mean readout   risk readout (c^T)                manip readout (c^J)')
for name,idx in (('B',P.kind[fit]=='safe'),('H',P.kind[fit]=='harmful'),('J',P.kind[fit]=='jailbreak')):
    print(f'  {name:3s}', np.round(r[idx].mean(0),2), np.round(m[idx].mean(0),2))
print('  W(B)', np.round(rw.mean(0),2), np.round(mw.mean(0),2))
y=(P.kind[fit]!='safe').astype(float)
for tag,(X_r,X_m,yy) in {'without W(B)':(r,m,y),'with W(B) as negatives':(np.vstack([r,rw]),np.vstack([m,mw]),np.r_[y,np.zeros(len(wb_tr))])}.items():
    X,fl,br=build_feature_matrix(X_r,X_m,[23],[14,15],[14,15,23])
    mdl=fit_sparse_fusion(X,yy,fl,config=FusionConfig())
    print(tag,'coef',dict(zip([f'{b[:4]}{l}' for b,l in zip(br,fl)],np.round(mdl.coef,3))),'converged',mdl.diagnostics['converged'])
# separability of J vs W(B) by manip readout (same wrapper, different goal)
jm=np.array([P.index[w['matched_to_jailbreak_id']] for w in [wb_rows[i] for i in wb_tr]])
_,mj=concept_scores(P.states[jm][:,L],bank)
print('AUROC(J vs its own W(B)) by readout, layers 14,15,23:  manip',[round(auroc(np.r_[np.ones(len(jm)),np.zeros(len(jm))],np.r_[mj[:,k],mw[:,k]]),2) for k in range(3)],
      ' risk',[round(auroc(np.r_[np.ones(len(jm)),np.zeros(len(jm))],np.r_[concept_scores(P.states[jm][:,L],bank)[0][:,k],rw[:,k]]),2) for k in range(3)])
