import sys, json, numpy as np
sys.path.insert(0,'tools/layer_screening'); sys.path.insert(0,'.')
from common import Panel, auroc
from stage2_matrices import build_bank
from stage2_length_controlled import load_controls
from dual_concept.concepts import concept_scores
from pathlib import Path
cache='/root/rivermind-data/dccf/cache'
panel=Panel('data/dccf_v1',cache)
crow,cstate=load_controls(Path('data/dccf_v1_length_controls'),Path(cache),panel.protocol)
cid={r['id']:i for i,r in enumerate(crow)}
cand=list(range(1,28)); li=[l-1 for l in cand]
tg={g for g,s,k in zip(panel.group,panel.split,panel.kind) if s=='train' and k=='harmful'}
bank,_,_=build_bank(panel,tg,li)
# padding effect: same H with vs without filler (paired), per layer; high AUROC means the readout detects "filler present"
J=[i for i in np.flatnonzero((panel.kind=='jailbreak')&(panel.split=='train')) if f"{panel.rows[i]['id']}__pad_h" in cid]
ph=np.array([cid[f"{panel.rows[i]['id']}__pad_h"] for i in J]); h=np.array([panel.index[f"{panel.group[i]}_h"] for i in J])
_,m_pad=concept_scores(cstate[np.ix_(ph,li)],bank); _,m_plain=concept_scores(panel.states[np.ix_(h,li)],bank)
r_pad,_=concept_scores(cstate[np.ix_(ph,li)],bank); r_plain,_=concept_scores(panel.states[np.ix_(h,li)],bank)
lab=np.r_[np.ones(len(J)),np.zeros(len(J))]
print('layer  manip: AUROC(H_pad vs H_plain)  mean shift | risk: AUROC(H_pad vs H_plain) mean shift')
for k,l in enumerate(cand):
    print(f'{l:3d}   {auroc(lab,np.r_[m_pad[:,k],m_plain[:,k]]):.2f}  {np.mean(m_pad[:,k]-m_plain[:,k]):+.3f}   |  {auroc(lab,np.r_[r_pad[:,k],r_plain[:,k]]):.2f}  {np.mean(r_pad[:,k]-r_plain[:,k]):+.3f}')
