# Q1 Allocation Solver - Math Modeling Competition 2023-C
import pandas as pd, numpy as np, random, time, os, json
from scipy.spatial.distance import cdist
warnings.filterwarnings('ignore')

# Parameters
N_WORKS, N_EXPERTS, K_PER_WORK = 3000, 125, 5
L_MIN, L_MAX = 115, 125
DELTA_SCORE, MATCH_TH = 8, 0.5
W_BAL, W_DIV, W_CHEAT = 0.4, 0.3, 0.3

def gen_data():
    np.random.seed(42); random.seed(42)
    exp_feat = np.random.dirichlet(np.ones(10), size=N_EXPERTS)
    work_feat = np.random.dirichlet(np.ones(10), size=N_WORKS)
    exp_bias = np.random.choice([-15,-5,0,5,15], size=N_EXPERTS, p=[.1,.2,.4,.2,.1])
    work_qual = np.clip(np.random.normal(70,15,N_WORKS), 0, 100)
    return exp_feat, work_feat, exp_bias, work_qual

def match_score(e_feat, w_feat):
    return 1 - cdist(e_feat.reshape(1,-1), w_feat.reshape(1,-1), 'cosine')[0,0]

def gen_scores(qual, bias):
    s = qual[:,None] + bias[None,:] + np.random.normal(0,5,(N_WORKS,N_EXPERTS))
    return np.clip(s, 0, 100)

def greedy_alloc(exp_feat, work_feat, qual, bias):
    alloc = np.zeros((N_WORKS,N_EXPERTS), dtype=int)
    load = np.zeros(N_EXPERTS, dtype=int)
    match_mat = np.array([[match_score(exp_feat[j], work_feat[i]) for j in range(N_EXPERTS)] for i in range(N_WORKS)])
    for i in range(N_WORKS):
        cands = [(j, match_mat[i,j]*0.7 + (1-load[j]/L_MAX)*0.3) for j in range(N_EXPERTS) if load[j]<L_MAX and match_mat[i,j]>=MATCH_TH]
        cands.sort(key=lambda x:-x[1])
        for j,_ in cands[:K_PER_WORK]:
            alloc[i,j]=1; load[j]+=1
    return alloc, load

def compute_obj(alloc, load, scores=None):
    bal = np.var(load)
    div = cheat = 0
    if scores is not None:
        for i in range(N_WORKS):
            ass = np.where(alloc[i]==1)[0]
            if len(ass)>=2:
                s=scores[i,ass]; rn=(s.max()-s.min())/100 if s.max()>s.min() else 0; div+=1-rn
                for a in range(len(ass)):
                    for b in range(a+1,len(ass)):
                        if abs(s[a]-s[b])<DELTA_SCORE: cheat+=1
        div/=N_WORKS
    return W_BAL*bal + W_DIV*div + W_CHEAT*cheat

def local_search(alloc, load, scores=None, swaps=200):
    best_a, best_l, best_o = alloc.copy(), load.copy(), compute_objective(alloc, load, scores)
    for _ in range(swaps):
        i1,i2 = np.random.choice(N_WORKS,2,replace=False)
        e1,e2 = np.where(best_a[i1]==1)[0], np.where(best_a[i2]==1)[0]
        if len(e1)<K_PER_WORK or len(e2)<K_PER_WORK: continue
        j1,j2 = np.random.choice(e1), np.random.choice(e2)
        new_a = best_a.copy()
        new_a[i1,j1],new_a[i1,j2] = new_a[i1,j2],new_a[i1,j1]
        new_a[i2,j1],new_a[i2,j2] = new_a[i2,j2],new_a[i2,j1]
        new_l = new_a.sum(axis=0)
        if np.any(new_l<L_MIN) or np.any(new_l>L_MAX): continue
        new_o = compute_obj(new_a, new_l, scores)
        if new_o < best_o: best_a,best_l,best_o = new_a,new_l,new_o
    return best_a, best_l, best_o

def evaluate(alloc, load, scores=None):
    m = {'balance': 1-np.std(load)/np.mean(load) if np.mean(load)>0 else 0}
    if scores is not None:
        divs=[]; valid=0
        for i in range(N_WORKS):
            ass=np.where(alloc[i]==1)[0]
            if len(ass)>=2:
                s=scores[i,ass]; rn=(s.max()-s.min())/100; divs.append(1-rn)
                md=min(abs(s[a]-s[b]) for a in range(len(s)) for b in range(a+1,len(s)))
                if md>=DELTA_SCORE: valid+=1
        m['diversity']=np.mean(divs) if divs else 0; m['anti_cheat']=valid/N_WORKS
    return m

def main():
    print('Q1 Allocation Solver'); t0=time.time()
    exp_f, work_f, exp_b, work_q = gen_data()
    scores = gen_scores(work_q, exp_b)
    print('Greedy init...'); alloc, load = greedy_alloc(exp_f, work_f, work_q, exp_b)
    print('Local search...'); alloc, load, obj = local_search(alloc, load, scores)
    metrics = evaluate(alloc, load, scores)
    os.makedirs(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results', exist_ok=True)
    rows = [{'work_id':i+1,'expert_id':j+1} for i in range(N_WORKS) for j in np.where(alloc[i]==1)[0]]
    pd.DataFrame(rows).to_csv(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q1_allocation.csv', index=False)
    pd.DataFrame([metrics]).to_csv(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q1_metrics.csv', index=False)
    with open(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q1_reproduce.json','w',encoding='utf-8') as f:
        json.dump({'seed':42,'params':{'N':N_WORKS,'M':N_EXPERTS,'K':K_PER_WORK},'metrics':metrics,'time':time.time()-t0},f,indent=2)
    print(f'Done in {time.time()-t0:.1f}s | Balance:{metrics[\
balance\]:.3f} Diversity:{metrics.get(\diversity\,0):.3f} AntiCheat:{metrics.get(\anti_cheat\,0)*100:.1f}%')

if __name__=='__main__': main()

