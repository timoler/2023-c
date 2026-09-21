import pandas as pd, numpy as np, random, time, os, json
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

N_W, N_E, K = 3000, 125, 5
L_MIN, L_MAX = 115, 125
DELTA, MATCH_TH = 8, 0.5
W_BAL, W_DIV, W_CHEAT = 0.4, 0.3, 0.3

def gen_data():
    np.random.seed(42); random.seed(42)
    ef = np.random.dirichlet(np.ones(10), size=N_E)
    wf = np.random.dirichlet(np.ones(10), size=N_W)
    eb = np.random.choice([-15,-5,0,5,15], size=N_E, p=[.1,.2,.4,.2,.1])
    wq = np.clip(np.random.normal(70,15,N_W), 0, 100)
    return ef, wf, eb, wq

def match(e, w):
    return 1 - cdist(e.reshape(1,-1), w.reshape(1,-1), 'cosine')[0,0]

def scores(q, b):
    s = q[:,None] + b[None,:] + np.random.normal(0,5,(N_W,N_E))
    return np.clip(s, 0, 100)

def greedy(ef, wf, q, b):
    alloc = np.zeros((N_W,N_E), dtype=int)
    load = np.zeros(N_E, dtype=int)
    mm = np.array([[match(ef[j], wf[i]) for j in range(N_E)] for i in range(N_W)])
    for i in range(N_W):
        cands = [(j, mm[i,j]*0.7 + (1-load[j]/L_MAX)*0.3) for j in range(N_E) if load[j]<L_MAX and mm[i,j]>=MATCH_TH]
        cands.sort(key=lambda x:-x[1])
        for j,_ in cands[:K]:
            alloc[i,j]=1; load[j]+=1
    return alloc, load

def obj(alloc, load, sc=None):
    bal = np.var(load)
    div = cheat = 0
    if sc is not None:
        for i in range(N_W):
            ass = np.where(alloc[i]==1)[0]
            if len(ass)>=2:
                s=sc[i,ass]; rn=(s.max()-s.min())/100 if s.max()>s.min() else 0; div+=1-rn
                for a in range(len(ass)):
                    for bb in range(a+1,len(ass)):
                        if abs(s[a]-s[bb])<DELTA: cheat+=1
        div/=N_W
    return W_BAL*bal + W_DIV*div + W_CHEAT*cheat

def local(alloc, load, sc=None, sw=200):
    ba,bl,bo = alloc.copy(), load.copy(), obj(alloc, load, sc)
    for _ in range(sw):
        i1,i2 = np.random.choice(N_W,2,replace=False)
        e1,e2 = np.where(ba[i1]==1)[0], np.where(ba[i2]==1)[0]
        if len(e1)<K or len(e2)<K: continue
        j1,j2 = np.random.choice(e1), np.random.choice(e2)
        na = ba.copy()
        na[i1,j1],na[i1,j2] = na[i1,j2],na[i1,j1]
        na[i2,j1],na[i2,j2] = na[i2,j2],na[i2,j1]
        nl = na.sum(axis=0)
        if np.any(nl<L_MIN) or np.any(nl>L_MAX): continue
        no = obj(na, nl, sc)
        if no < bo: ba,bl,bo = na,nl,no
    return ba, bl, bo

def eval_m(alloc, load, sc=None):
    m = {'balance': 1-np.std(load)/np.mean(load) if np.mean(load)>0 else 0}
    if sc is not None:
        divs=[]; valid=0
        for i in range(N_W):
            ass=np.where(alloc[i]==1)[0]
            if len(ass)>=2:
                s=sc[i,ass]; rn=(s.max()-s.min())/100; divs.append(1-rn)
                md=min(abs(s[a]-s[b]) for a in range(len(s)) for b in range(a+1,len(s)))
                if md>=DELTA: valid+=1
        m['diversity']=np.mean(divs) if divs else 0; m['anti_cheat']=valid/N_W
    return m

print('Q1 Solver starting...'); t0=time.time()
ef, wf, eb, wq = gen_data()
sc = scores(wq, eb)
print('Greedy alloc...'); alloc, load = greedy(ef, wf, wq, eb)
print('Local search...'); alloc, load, o = local(alloc, load, sc)
metrics = eval_m(alloc, load, sc)
out = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results'
os.makedirs(out, exist_ok=True)
rows = [{'work_id':i+1,'expert_id':j+1} for i in range(N_W) for j in np.where(alloc[i]==1)[0]]
pd.DataFrame(rows).to_csv(out+'/q1_alloc.csv', index=False)
pd.DataFrame([metrics]).to_csv(out+'/q1_metrics.csv', index=False)
with open(out+'/q1_repro.json','w') as f: json.dump({'seed':42,'metrics':metrics,'time':time.time()-t0},f,indent=2)
print('Done in', round(time.time()-t0,1), 's')
print('Balance:', round(metrics['balance'],3))
print('Diversity:', round(metrics['diversity'],3))
print('AntiCheat:', round(metrics['anti_cheat']*100,1), '%')

