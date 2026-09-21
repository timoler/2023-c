import pandas as pd, numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

fp = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx'
df = pd.read_excel(fp, engine='openpyxl', header=1)
cols = df.columns[0:5].tolist()

res = []
for i, r in df.iterrows():
    sc = [pd.to_numeric(r[c], errors='coerce') for c in cols]
    sc = [s for s in sc if pd.notna(s)]
    if len(sc) < 2: continue
    a = np.array(sc)
    z = (a - a.mean()) / a.std() if a.std() > 1e-10 else np.zeros_like(a)
    res.append({'id': i, 'std_score': z.mean(), 'n': len(sc)})

df2 = pd.DataFrame(res)
print('Count:', len(df2))
print('Range:', df2['std_score'].min(), 'to', df2['std_score'].max())
print(df2.nlargest(5, 'std_score')[['id','std_score']].to_string(index=False))

out = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_scores.csv'
df2.to_csv(out, index=False)
print('Saved:', out)
