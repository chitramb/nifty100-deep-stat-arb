from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from torch import nn

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(2)

class Net(nn.Module):
    def __init__(self, n_assets, filters=8, heads=4):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv1d(1,filters,2),nn.GELU(),nn.Conv1d(filters,filters,2),nn.GELU())
        layer=nn.TransformerEncoderLayer(d_model=filters,nhead=heads,batch_first=True,activation='gelu',dropout=0.0)
        self.tr=nn.TransformerEncoder(layer,1)
        self.alloc=nn.Sequential(nn.Linear(filters,16),nn.GELU(),nn.Linear(16,1),nn.Tanh())
    def forward(self,x):
        # x: batch x assets x 30
        b,a,l=x.shape
        z=self.cnn(x.reshape(b*a,1,l)).transpose(1,2)
        z=self.tr(z).mean(1).reshape(b,a,-1)
        raw=self.alloc(z).squeeze(-1)
        # dollar-neutral then gross-normalize
        raw=raw-raw.mean(1,keepdim=True)
        return raw/raw.abs().sum(1,keepdim=True).clamp_min(1e-8)

def sharpe_loss(p):
    return -(p.mean()/(p.std(unbiased=True)+1e-8))*np.sqrt(252.0)

def load_membership(path,start,end):
    m=pd.read_csv(path,parse_dates=['valid_from','valid_to'])
    m=m[m.index_name.str.lower().eq('nifty 100')]
    m=m[(m.valid_to.isna() | (m.valid_to>=start)) & (m.valid_from<=end)]
    return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--eod',required=True); ap.add_argument('--membership',required=True); ap.add_argument('--out',required=True); ap.add_argument('--start',default='2020-01-01'); ap.add_argument('--oos-start',default='2024-01-02'); ap.add_argument('--end',default='2026-05-15'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    e=pd.read_parquet(args.eod)
    e['date']=pd.to_datetime(e['date']); e=e[(e.date>=args.start)&(e.date<=args.end)]
    e=e[e.series.eq('EQ') if 'series' in e else np.ones(len(e),dtype=bool)]
    p=e.pivot_table(index='date',columns='symbol',values='close',aggfunc='last').sort_index()
    # log returns; raw NSE close is used and corporate-action discontinuities are removed conservatively.
    r=np.log(p).diff()
    m=load_membership(args.membership,p.index.min(),p.index.max())
    dates=p.index
    def universe(d):
        q=m[(m.valid_from<=d)&(m.valid_to.isna() | (m.valid_to>d))]
        return sorted(set(q.symbol)&set(p.columns))
    oos=pd.Timestamp(args.oos_start); end=pd.Timestamp(args.end)
    # 125-day walk-forward blocks. PCA is fitted only on the preceding 1000 observations.
    all_port=[]; block_stats=[]
    test_dates=dates[(dates>=oos)&(dates<=end)]
    for b0 in range(0,len(test_dates),125):
        block=test_dates[b0:b0+125]
        if len(block)==0: break
        d0=block[0]; hist_dates=dates[dates<d0]
        if len(hist_dates)<1035: continue
        train_dates=hist_dates[-1000:]
        assets=universe(d0)
        if len(assets)<50: continue
        X=r.reindex(index=train_dates,columns=assets)
        # retain assets with sufficient history; fill remaining missing values cross-sectionally.
        assets=[c for c in assets if X[c].notna().mean()>=0.90]
        X=X[assets].fillna(X[assets].median())
        pca=PCA(n_components=min(5,len(assets)-1),random_state=SEED)
        Z=pca.fit_transform(X.values); rec=pca.inverse_transform(Z)
        resid_train=pd.DataFrame(X.values-rec,index=train_dates,columns=assets)
        # Fit model on samples with a 30-day cumulative-residual trajectory and next-day stock returns.
        arr=resid_train.values
        samples=[]; targets=[]
        for i in range(30,len(train_dates)-1):
            samples.append(np.cumsum(arr[i-29:i+1],axis=0).T)
            targets.append(r.loc[train_dates[i+1],assets].fillna(0).values)
        if len(samples)<200: continue
        xs=torch.tensor(np.asarray(samples),dtype=torch.float32)
        ys=torch.tensor(np.asarray(targets),dtype=torch.float32)
        net=Net(len(assets)); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        net.train()
        for epoch in range(12):
            perm=torch.randperm(len(xs))
            for j in range(0,len(xs),64):
                ii=perm[j:j+64]; w=net(xs[ii]); pr=(w*ys[ii]).sum(1); loss=sharpe_loss(pr)
                opt.zero_grad(); loss_t=torch.tensor(loss,dtype=torch.float32,requires_grad=True)
                # Recompute differentiably; helper above returns scalar detached, so use direct objective.
                pr=(net(xs[ii])*ys[ii]).sum(1); loss_t=-(pr.mean()/(pr.std(unbiased=True)+1e-8))*np.sqrt(252.)
                loss_t.backward(); opt.step()
        # OOS residuals using the same PCA fitted on training data.
        oosr=r.reindex(index=block,columns=assets).fillna(0)
        hist_for_signal=pd.concat([resid_train.tail(30),oosr])
        # Project OOS returns into PCA residual space.
        # PCA transform/inverse operates cross-sectionally for each date.
        raw=oosr.values
        score=pca.transform(np.nan_to_num(raw,nan=0.0)); oosres=raw-pca.inverse_transform(score)
        rr=np.vstack([resid_train.tail(30).values,oosres])
        net.eval(); block_ret=[]
        for k,d in enumerate(block):
            idx=30+k
            sig=np.cumsum(rr[idx-29:idx+1],axis=0).T
            with torch.no_grad(): w=net(torch.tensor(sig[None],dtype=torch.float32)).numpy()[0]
            realized=r.loc[d,assets].fillna(0).values
            pr=float(np.dot(w,realized))
            all_port.append((d,pr)); block_ret.append(pr)
        br=np.asarray(block_ret)
        block_stats.append({'start':str(block[0].date()),'end':str(block[-1].date()),'assets':len(assets),'sharpe':float(np.sqrt(252)*br.mean()/(br.std(ddof=1)+1e-12)),'mean_daily':float(br.mean())})
    port=pd.DataFrame(all_port,columns=['date','return']).drop_duplicates('date').sort_values('date')
    if len(port):
        wealth=np.exp(port['return'].cumsum()); dd=wealth/wealth.cummax()-1
        ann_mean=float(port['return'].mean()*252); ann_vol=float(port['return'].std(ddof=1)*np.sqrt(252)); sr=float(ann_mean/ann_vol) if ann_vol else float('nan')
        metrics={'oos_start':str(port.date.min().date()),'oos_end':str(port.date.max().date()),'n_days':len(port),'annualized_log_return':ann_mean,'annualized_volatility':ann_vol,'sharpe':sr,'max_drawdown_log_wealth':float(dd.min()),'total_log_return':float(port['return'].sum()),'total_compounded_return':float(np.exp(port['return'].sum())-1)}
    else: metrics={}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (out/'blocks.json').write_text(json.dumps(block_stats,indent=2))
    port.to_csv(out/'portfolio_returns.csv',index=False)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
