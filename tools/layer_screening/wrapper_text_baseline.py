import sys, json, numpy as np
sys.path.insert(0,'tools/layer_screening'); sys.path.insert(0,'.')
from common import Panel, auroc
from dual_concept.backbone import StateCache
from stage2_length_controlled import load_controls
from stage5_nested_eval import load_flat
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
cache='/root/rivermind-data/dccf/cache_v2'
P=Panel('data/dccf_v1',cache); C=StateCache(Path(cache)/'rows')
crow,_=load_controls(Path('data/dccf_v1_length_controls'),Path(cache),P.protocol); cby={r['id']:r for r in crow}
wb,_=load_flat(Path('data/dccf_v1_wrapper_controls/development'),['wrapped_benign'],C,P.protocol)
keep=[w for w in wb if w['matched_to_jailbreak_id']+'__pad_b' in cby]
txt_w=[w['text'] for w in keep]; txt_p=[cby[w['matched_to_jailbreak_id']+'__pad_b']['text'] for w in keep]
y=np.r_[np.ones(len(keep)),np.zeros(len(keep))]; T=txt_w+txt_p; sem=np.array([w['semantic_strategy'] for w in keep]*2)
print('n pairs',len(keep),'| token counts equal within 3 by construction')
print('length-only AUROC(W(B) vs B_pad):',round(auroc(y,np.array([len(t) for t in T],float)),3))
res={}
for s in sorted(set(sem)):
    tr=sem!=s; te=sem==s
    v=CountVectorizer(min_df=2); X=v.fit_transform([t for t,m in zip(T,tr) if m])
    lr=LogisticRegression(max_iter=2000).fit(X,y[tr]); res[str(s)]=round(auroc(y[te],lr.decision_function(v.transform([t for t,m in zip(T,te) if m]))),3)
print('leave-one-strategy-out bag-of-words, W(B) vs B_pad:',res)
