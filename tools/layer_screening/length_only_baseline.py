import sys, json, numpy as np
sys.path.insert(0,'tools/layer_screening'); sys.path.insert(0,'.')
from common import Panel, auroc
cache='/root/rivermind-data/dccf/cache_v2'
p=Panel('data/dccf_v1',cache)
tok=np.array([r['full_input_token_count'] for r in p.rows],float)
sel=p.split=='select'
for name,mask_pos in (('jailbreak',p.kind=='jailbreak'),('harmful',p.kind=='harmful')):
    m=sel&(mask_pos|(p.kind=='safe'))
    lab=(p.kind[m]!='safe').astype(int)
    print(f'select: token-count-only AUROC({name} vs safe B) = {auroc(lab,tok[m]):.3f}')
print('select median tokens  safe %.0f  harmful %.0f  jailbreak %.0f'%tuple(np.median(tok[sel&(p.kind==k)]) for k in('safe','harmful','jailbreak')))
# fixed-threshold length rule calibrated like the detector: 5% of calibrate requests above it
cal=[json.loads(l) for l in open('data/dccf_v1/development/calibrate.jsonl')]
ct=np.array([r['full_input_token_count'] for r in cal],float); thr=np.sort(ct)[::-1][10]+1e-9
j=sel&(p.kind=='jailbreak'); h=sel&(p.kind=='harmful'); b=sel&(p.kind=='safe')
print(f'length-only rule (calibrate 5% budget, thr={thr:.0f} tokens): select jailbreak TPR {np.mean(tok[j]>=thr):.2f}  harmful TPR {np.mean(tok[h]>=thr):.2f}  safe B FPR {np.mean(tok[b]>=thr):.2f}')
