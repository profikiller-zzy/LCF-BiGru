import sys, json, csv, numpy as np
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
rows=[]
for held in sorted(set(panel.family[(panel.kind=='jailbreak')&(panel.split=='train')])):
    J=np.array([i for i in np.flatnonzero((panel.kind=='jailbreak')&(panel.split=='train')&(panel.family==held)) if f"{panel.rows[i]['id']}__pad_h" in cid])
    hg={panel.group[i] for i in J}; fit=tg-hg
    bank,_,_=build_bank(panel,fit,li)
    # padding shift estimated ONLY on held-in groups: mean(m(H_pad) - m(H_plain)) per layer
    fj=np.array([i for i in np.flatnonzero((panel.kind=='jailbreak')&(panel.split=='train')&np.isin(panel.group,list(fit))) if f"{panel.rows[i]['id']}__pad_h" in cid])
    fh=np.array([panel.index[f"{panel.group[i]}_h"] for i in fj]); fp=np.array([cid[f"{panel.rows[i]['id']}__pad_h"] for i in fj])
    _,mp=concept_scores(cstate[np.ix_(fp,li)],bank); _,mh=concept_scores(panel.states[np.ix_(fh,li)],bank)
    shift=(mp-mh).mean(0); n_tok=np.array([panel.rows[i]['full_input_token_count'] for i in fj])
    # J compared with the UNPADDED exact H, after subtracting the shift that padding alone causes
    _,mj=concept_scores(panel.states[np.ix_(J,li)],bank)
    hh=np.array([panel.index[f"{panel.group[i]}_h"] for i in J]); _,mhh=concept_scores(panel.states[np.ix_(hh,li)],bank)
    ph=np.array([cid[f"{panel.rows[i]['id']}__pad_h"] for i in J]); _,mph=concept_scores(cstate[np.ix_(ph,li)],bank)
    lab=np.r_[np.ones(len(J)),np.zeros(len(J))]
    for k,l in enumerate(cand):
        rows.append({'family':held,'layer':l,'n_J':len(J),'auroc_vs_pad_h':auroc(lab,np.r_[mj[:,k],mph[:,k]]),
                     'auroc_vs_plain_h_shift_removed':auroc(lab,np.r_[mj[:,k]-shift[k],mhh[:,k]]),
                     'delta_vs_pad':float((mj[:,k]-mph[:,k]).mean()),'padding_shift':float(shift[k])})
    print(held,'done',flush=True)
out=Path('/root/rivermind-data/dccf/runs/layer_screening_v2/stage2_length_controlled'); 
with (out/'padding_shift_check.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
fams=sorted({r['family'] for r in rows})
for key in('auroc_vs_pad_h','delta_vs_pad','padding_shift'):
    print('\n',key)
    for fm in fams: print(f'{fm:9s}',' '.join(f"{[r for r in rows if r['family']==fm and r['layer']==l][0][key]:.2f}" for l in (9,13,14,17,20,23,25,27)))
