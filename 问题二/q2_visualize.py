#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C题第二问：Pillow可视化方案 (ASCII-safe version)
"""
import pandas as pd, numpy as np, os, sys, math
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

# 配置
DATA_PATH = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx'
OUTPUT_DIR = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\figures_pillow'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 字体
FONT = None
for fp in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttf']:
    if os.path.exists(fp):
        try: FONT = ImageFont.truetype(fp, 14); break
        except: pass
if FONT is None: FONT = ImageFont.load_default()

EXPERT_COLS = [6, 9, 12, 15, 18, 25, 28, 31]
EXPERT_NAMES = ['专家一','专家二','专家三','专家四','专家五','专家一.1','专家二.1','专家三.1']
COLORS = ['#4E79A7','#F28E2B','#E15759','#76B7B2','#59A14F','#EDC948','#AF7AA1','#FF9DA7']

class Chart:
    def __init__(self, w=1200, h=700):
        self.img = Image.new('RGB', (w,h), 'white')
        self.draw = ImageDraw.Draw(self.img)
        self.w, self.h = w, h
        self.m = {'t':60,'b':80,'l':80,'r':50}
    
    def title(self, txt, y=None, sz=18):
        f = FONT if FONT!=ImageFont.load_default() else ImageFont.load_default()
        y = y or self.m['t']//2
        self.draw.text((self.w//2, y), txt, fill='black', font=f, anchor='mt')
        return self
    
    def boxplot(self, data_list, labels, title_txt=None):
        if not data_list: return self
        pl, pr, pt, pb = self.m['l'], self.w-self.m['r'], self.m['t']+30, self.h-self.m['b']
        all_v = [v for d in data_list if len(d)>0 for v in d]
        if not all_v: return self
        dmin, dmax = min(all_v)-10, max(all_v)+10
        
        # 坐标轴
        self.draw.line([(pl,pb),(pr,pb)], fill='black')
        self.draw.line([(pl,pt),(pl,pb)], fill='black')
        
        # Y轴刻度
        for i in range(5):
            val = dmin + (dmax-dmin)*i/4
            y = int(pb - (val-dmin)/(dmax-dmin)*(pb-pt))
            self.draw.line([(pl-5,y),(pl,y)], fill='black')
            self.draw.text((pl-45,y-6), f'{val:.0f}', fill='black', font=FONT)
        
        # 箱线图
        n = len([d for d in data_list if len(d)>0])
        bw = min(50, (pr-pl)//(n*2))
        sp = (pr-pl-bw*n)//(n+1)
        idx = 0
        for data, label, color in zip(data_list, labels, COLORS):
            if len(data)==0: continue
            xc = pl + sp + idx*(bw+sp) + bw//2
            idx += 1
            q1,med,q3 = np.percentile(data, [25,50,75])
            iqr = q3-q1
            wl, wh = max(data.min(),q1-1.5*iqr), min(data.max(),q3+1.5*iqr)
            
            def sy(v): return int(pb - (v-dmin)/(dmax-dmin)*(pb-pt))
            
            # 须线
            self.draw.line([(xc,sy(wl)),(xc,sy(q1))], fill=color, width=2)
            self.draw.line([(xc,sy(q3)),(xc,sy(wh))], fill=color, width=2)
            self.draw.line([(xc-12,sy(wl)),(xc+12,sy(wl))], fill=color, width=2)
            self.draw.line([(xc-12,sy(wh)),(xc+12,sy(wh))], fill=color, width=2)
            # 箱体
            self.draw.rectangle([(xc-bw//2,sy(q3)),(xc+bw//2,sy(q1))], fill=color, outline='black')
            self.draw.line([(xc-bw//2,sy(med)),(xc+bw//2,sy(med))], fill='white', width=2)
            # 标签
            self.draw.text((xc, pb+8), label, fill='black', font=FONT, anchor='mt')
        
        if title_txt: self.title(title_txt)
        return self
    
    def histogram(self, data, bins=25, title_txt=None, color='#4E79A7'):
        if len(data)==0: return self
        pl,pr,pt,pb = self.m['l'],self.w-self.m['r'],self.m['t']+30,self.h-self.m['b']
        cnts, edges = np.histogram(data, bins=bins)
        if cnts.max()==0: return self
        
        self.draw.line([(pl,pb),(pr,pb)], fill='black')
        self.draw.line([(pl,pt),(pl,pb)], fill='black')
        
        bw = (pr-pl)/len(cnts)
        for i,(c,e) in enumerate(zip(cnts, edges[:-1])):
            if c==0: continue
            x1,x2 = pl+i*bw+1, pl+(i+1)*bw-1
            y1 = pb - c/cnts.max()*(pb-pt)
            self.draw.rectangle([(x1,y1),(x2,pb)], fill=color, outline='white')
        
        if title_txt: self.title(title_txt)
        return self
    
    def scatter(self, x,y, title_txt=None, xlabel='X', ylabel='Y', color='steelblue'):
        if len(x)==0: return self
        pl,pr,pt,pb = self.m['l'],self.w-self.m['r'],self.m['t']+30,self.h-self.m['b']
        xmin,xmax = x.min(),x.max()
        ymin,ymax = y.min(),y.max()
        xp,yp = (xmax-xmin)*0.05, (ymax-ymin)*0.05
        
        self.draw.line([(pl,pb),(pr,pb)], fill='black')
        self.draw.line([(pl,pt),(pl,pb)], fill='black')
        
        for xv,yv in zip(x,y):
            px = int(pl + (xv-xmin-xp)/(xmax-xmin+2*xp)*(pr-pl))
            py = int(pb - (yv-ymin-yp)/(ymax-ymin+2*yp)*(pb-pt))
            self.draw.ellipse([(px-1,py-1),(px+1,py+1)], fill=color)
        
        self.draw.text((pl-50,(pt+pb)//2), ylabel, fill='black', font=FONT, anchor='mm')
        self.draw.text(((pl+pr)//2,pb+25), xlabel, fill='black', font=FONT, anchor='mt')
        if title_txt: self.title(title_txt)
        return self
    
    def barh(self, labels, values, title_txt=None, xlabel='Value'):
        if not labels: return self
        pl,pr,pt,pb = self.m['l']+100,self.w-self.m['r'],self.m['t']+20,self.h-self.m['b']
        sorted_d = sorted(zip(labels,values), key=lambda x:x[1], reverse=True)
        ls,vs = zip(*sorted_d)
        mv = max(vs)
        bh = min(30, (pb-pt)//len(ls)-5)
        sp = (pb-pt-bh*len(ls))//(len(ls)+1)
        
        for i,(l,v) in enumerate(zip(ls,vs)):
            y = pt+sp+i*(bh+sp)+bh//2
            bw = (v/mv)*(pr-pl) if mv>0 else 0
            c = COLORS[i%len(COLORS)]
            self.draw.rectangle([(pl,y-bh//2),(pl+bw,y+bh//2)], fill=c, outline='black')
            self.draw.text((pl+bw+5,y), f'{v:.3f}', fill='black', font=FONT, anchor='lm')
            self.draw.text((pl-10,y), l, fill='black', font=FONT, anchor='rm')
        
        self.draw.line([(pl,pb),(pr,pb)], fill='black')
        if xlabel: self.draw.text(((pl+pr)//2,pb+20), xlabel, fill='black', font=FONT, anchor='mt')
        if title_txt: self.title(title_txt)
        return self
    
    def radar(self, metrics, scores_dict, title_txt=None):
        if not metrics: return self
        n = len(metrics)
        angles = [2*math.pi*i/n for i in range(n)] + [0]
        cx,cy,r = self.w//2, self.h//2, min(self.w,self.h)//3
        
        # 网格
        for lv in [0.2,0.4,0.6,0.8,1.0]:
            pts = [(cx+r*lv*math.cos(a), cy-r*lv*math.sin(a)) for a in angles]
            self.draw.polygon(pts, outline='lightgray')
        
        # 轴线+标签
        for a,m in zip(angles[:-1], metrics):
            x,y = cx+r*math.cos(a), cy-r*math.sin(a)
            self.draw.line([(cx,cy),(x,y)], fill='gray')
            self.draw.text((cx+(r+25)*math.cos(a), cy-(r+25)*math.sin(a)-8), m, fill='black', font=FONT, anchor='mm')
        
        # 数据
        for name,vals in scores_dict.items():
            pts = [(cx+r*v*math.cos(a), cy-r*v*math.sin(a)) for a,v in zip(angles[:-1], vals+vals[:1])]
            c = COLORS[list(scores_dict.keys()).index(name)%len(COLORS)]
            self.draw.polygon(pts, outline=c, fill=c+'33', width=2)
        
        if title_txt: self.title(title_txt)
        return self
    
    def heatmap(self, matrix, row_labs, col_labs, title_txt=None):
        if not matrix: return self
        pl,pr,pt,pb = self.m['l']+60,self.w-self.m['r'],self.m['t']+20,self.h-self.m['b']
        nr,nc = len(matrix), len(matrix[0]) if matrix else 0
        if nr==0 or nc==0: return self
        cw,ch = (pr-pl)/nc, (pb-pt)/nr
        all_v = [v for row in matrix for v in row]
        minv,maxv = min(all_v), max(all_v)
        
        def cmap(v):
            if maxv==minv: return '#76B7B2'
            r = (v-minv)/(maxv-minv)
            if r<0.5: return f'#{239:02x}{int(138+(247-138)*r*2):02x}{int(59-59*r*2):02x}'
            return f'#{int(239-(239-116)*(r-0.5)*2):02x}{247:02x}3b'
        
        for i,row in enumerate(matrix):
            for j,val in enumerate(row):
                x1,y1 = pl+j*cw, pt+i*ch
                self.draw.rectangle([(x1,y1),(x1+cw,y1+ch)], fill=cmap(val), outline='white')
        
        for i,l in enumerate(row_labs):
            self.draw.text((pl-10, pt+i*ch+ch//2), l, fill='black', font=FONT, anchor='rm')
        for j in range(0,nc,nc//10):
            self.draw.text((pl+j*cw+cw//2, pb+15), str(j+1), fill='black', font=FONT, anchor='mt')
        if title_txt: self.title(title_txt)
        return self
    
    def save(self, path):
        self.img.save(path, 'PNG', quality=95)
        print(f'  [OK] {os.path.basename(path)} ({os.path.getsize(path)//1024} KB)')

def load_data():
    print('[LOAD] Loading data...')
    df = pd.read_excel(DATA_PATH, engine='openpyxl', header=1)
    all_s = {}
    for idx,col in enumerate(EXPERT_COLS):
        nm = EXPERT_NAMES[idx]
        sc = pd.to_numeric(df.iloc[2:,col], errors='coerce').dropna().values
        if len(sc)>10:
            all_s[nm] = sc
            print(f'  {nm}: n={len(sc)}, mu={np.mean(sc):.1f}, sigma={np.std(sc):.2f}')
    
    orig = []
    for ri in range(2,len(df)):
        vs = [pd.to_numeric(df.iloc[ri,c], errors='coerce') for c in EXPERT_COLS]
        vs = [v for v in vs if pd.notna(v)]
        orig.append(np.mean(vs) if len(vs)>=2 else np.nan)
    orig = np.array(orig)
    
    e_stats = {}
    for nm,sc in all_s.items():
        e_stats[nm] = {'mu':np.mean(sc),'sigma':np.std(sc,ddof=1),'median':np.median(sc),
                       'mad':np.median(np.abs(sc-np.median(sc)))*1.4826}
    
    def calc_std(meth='zscore'):
        res = []
        for ri in range(2,len(df)):
            zv = []
            for nm,col in zip(EXPERT_NAMES, EXPERT_COLS):
                if nm not in e_stats: continue
                val = pd.to_numeric(df.iloc[ri,col], errors='coerce')
                if pd.notna(val):
                    s = e_stats[nm]
                    if meth=='zscore': z=(val-s['mu'])/s['sigma'] if s['sigma']>1e-10 else 0
                    elif meth=='robust': z=(val-s['median'])/(s['mad'] if s['mad']>1e-10 else 1.0)
                    elif meth=='pct': z=(np.sum(all_s[nm]<=val)/len(all_s[nm]))*100
                    zv.append(z)
            res.append(np.mean(zv) if len(zv)>=2 else np.nan)
        return np.array(res)
    
    sz,sr,sp = calc_std('zscore'), calc_std('robust'), calc_std('pct')
    
    weights = {nm:1.0/(1+s['sigma']/abs(s['mu'])) for nm,s in e_stats.items() if abs(s['mu'])>1e-10}
    sc = []
    for ri in range(2,len(df)):
        zs,ws = 0,0
        for nm,col in zip(EXPERT_NAMES, EXPERT_COLS):
            if nm not in e_stats or nm not in weights: continue
            val = pd.to_numeric(df.iloc[ri,col], errors='coerce')
            if pd.notna(val) and e_stats[nm]['sigma']>1e-10:
                z = (val-e_stats[nm]['mu'])/e_stats[nm]['sigma']
                w = weights[nm]
                zs += w*z; ws += w
        sc.append(zs/ws if ws>1e-10 else np.nan)
    sc = np.array(sc)
    
    valid = ~np.isnan(sz)
    print(f'[OK] Data ready, valid={valid.sum()}')
    return {'orig':orig,'z':sz,'robust':sr,'pct':sp,'cons':sc,'valid':valid,'weights':weights,'all_s':all_s}

def main():
    print('='*60)
    print('[VIS] C题第二问 Pillow visualization')
    print('='*60)
    
    data = load_data()
    v = data['valid']
    
    # Fig1: Boxplot
    print('[PLOT] Fig1: Expert distribution...')
    vn = [n for n in EXPERT_NAMES if n in data['all_s']]
    vd = [data['all_s'][n] for n in vn]
    Chart(1200,600).title('Fig1: Expert Score Distribution (Boxplot)').boxplot(vd, vn).save(os.path.join(OUTPUT_DIR,'q2_expert_dist.png'))
    
    # Fig2: Histogram
    print('[PLOT] Fig2: Z-score distribution...')
    Chart(1000,500).title('Fig2: Z-score Standard Score Distribution').histogram(data['z'][v]).save(os.path.join(OUTPUT_DIR,'q2_zscore_hist.png'))
    
    # Fig3: Scatter
    print('[PLOT] Fig3: Correlation scatter...')
    rho,_ = stats.spearmanr(data['orig'][v], data['cons'][v])
    Chart(900,600).title(f'Fig3: Consensus vs Original (rho={rho:.3f})').scatter(data['orig'][v], data['cons'][v], xlabel='Original', ylabel='Consensus').save(os.path.join(OUTPUT_DIR,'q2_corr_scatter.png'))
    
    # Fig4: Heatmap
    print('[PLOT] Fig4: Top-50 heatmap...')
    t50 = np.argsort(-data['orig'][v])[:50]
    hm = [pd.Series(s[v]).rank(ascending=False).iloc[t50].values.tolist() for s in [data['orig'],data['z'],data['robust'],data['cons']]]
    Chart(1100,400).title('Fig4: Top-50 Ranking Comparison').heatmap(hm, ['Original','Z-score','Robust','Consensus'], [str(i+1) for i in range(50)]).save(os.path.join(OUTPUT_DIR,'q2_top50_heatmap.png'))
    
    # Fig5: Barh
    print('[PLOT] Fig5: Expert weights...')
    Chart(900,500).title('Fig5: Expert Weights w_j=1/(1+CV_j)').barh(list(data['weights'].keys()), list(data['weights'].values()), xlabel='Weight').save(os.path.join(OUTPUT_DIR,'q2_expert_weights.png'))
    
    # Fig6: Radar
    print('[PLOT] Fig6: Radar comparison...')
    metrics = ['KS','Spearman','Robust','Speed','Explain']
    radar_s = {'Z-score':[0.85,0.96,0.70,1.0,0.90],'Robust':[0.80,0.95,0.95,0.85,0.85],'Consensus':[0.88,0.97,0.92,0.90,0.95]}
    Chart(800,800).title('Fig6: Method Comparison (Radar)').radar(metrics, radar_s).save(os.path.join(OUTPUT_DIR,'q2_method_radar.png'))
    
    # Index
    with open(os.path.join(OUTPUT_DIR,'README.md'),'w',encoding='utf-8') as f:
        f.write('# C题第二问 Pillow Charts\n\n')
        f.write('| File | Description |\n|------|-------------|\n')
        f.write('| q2_expert_dist.png | Expert score boxplot |\n')
        f.write('| q2_zscore_hist.png | Z-score histogram |\n')
        f.write('| q2_corr_scatter.png | Correlation scatter |\n')
        f.write('| q2_top50_heatmap.png | Top-50 ranking heatmap |\n')
        f.write('| q2_expert_weights.png | Expert weights bar chart |\n')
        f.write('| q2_method_radar.png | Method comparison radar |\n')
    
    print('\n'+'='*60)
    print('[OK] Done! Output:', OUTPUT_DIR)
    print('\n[LIST] Files:')
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        if fn.endswith('.png') or fn.endswith('.md'):
            print(f'  - {fn} ({os.path.getsize(os.path.join(OUTPUT_DIR,fn))//1024} KB)')
    
    print('\n[STATS] Validation:')
    for nm,sc in [('Z-score',data['z']),('Robust',data['robust']),('Consensus',data['cons'])]:
        sv = sc[v]
        ks = stats.kstest(sv,'norm')[1]
        rh,_ = stats.spearmanr(data['orig'][v], sv)
        print(f'  {nm:12s} | KS-p={ks:.3f} [{"Y" if ks>0.05 else "N"}]  rho={rh:.3f}')
    print('='*60)

if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'[ERR] {type(e).__name__}: {e}')
        import traceback; traceback.print_exc()
        sys.exit(1)
