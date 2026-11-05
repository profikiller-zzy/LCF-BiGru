import sys, json, numpy as np
sys.path.insert(0,'tools/layer_screening'); sys.path.insert(0,'.')
from common import Panel
from stage2_matrices import build_bank
from dual_concept.concepts import concept_scores
from dual_concept.fusion import FusionConfig, build_feature_matrix, fit_sparse_fusion
cache='/root/rivermind-data/dccf/cache'
panel=Panel('data/dccf_v1',cache)
cand=list(range(1,28)); li=[l-1 for l in cand]
tg={g for g,s,k in zip(panel.group,panel.split,panel.kind) if s=='train' and k=='harmful'}
bank,_,_=build_bank(panel,tg,li)
rows=np.flatnonzero(panel.split=='train')
r,m=concept_scores(panel.states[np.ix_(rows,li)],bank); y=(panel.kind[rows]!='safe').astype(float)
for name,(rl,ml) in {'risk[25,15]+manip[14,15]':([25,15],[14,15]),'risk[25,15]':([25,15],[]),'manip[14,15]':([],[14,15])}.items():
    rl_=rl or [1]; ml_=ml or [1]
    X,fl,br=build_feature_matrix(r,m,rl_,ml_,cand)
    keep=np.r_[np.ones(len(rl_),bool) if rl else np.zeros(len(rl_),bool), np.ones(len(ml_),bool) if ml else np.zeros(len(ml_),bool)]
    X,fl,br=X[:,keep],fl[keep],br[keep]
    mdl=fit_sparse_fusion(X,y,fl,config=FusionConfig())
    print(name,[(b,int(l),round(float(w),3)) for b,l,w in zip(br,fl,mdl.coef)],'converged',mdl.diagnostics['converged'])
