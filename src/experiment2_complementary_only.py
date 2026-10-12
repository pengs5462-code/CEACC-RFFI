#!/usr/bin/env python3
"""Experiment 2: Fixed-AST selected residual versus retrained complementary-only."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np, torch
from calibrated_rejection_fusion_multiknown import loo_cdf,reference_cdf
from continuous_gcr_loeo import quantile
from evaluate_new_arch_gcr_matched import common_features,metric_row,oscr,write_csv
from experiment1_fixed_ast import SPLITS,SEEDS,LAMBDAS,fixed_map,fold_features
from fisher_retaining_cascade import BETAS,build_retaining_scores,bootstrap_ci

CANDIDATE='nophase_alpha0'

def inner(root,out,batch,device,worker,workers):
 rows=[]; tasks=[]
 data=root/'dataset/multiknown17_v1'
 for s in SPLITS:
  known=list(map(int,json.loads((data/f'split_{s}.json').read_text())['known_classes']))
  for seed in SEEDS:
   for held in known: tasks.append((s,seed,held))
 for i,(s,seed,held) in enumerate(tasks):
  if i%workers!=worker: continue
  f,isk,correct=fold_features(root,s,seed,held,CANDIDATE,batch,device)
  for lam in LAMBDAS:
   vr=f['val_sh']+lam*f['val_sc']; ur=f['unk_sh']+lam*f['unk_sc']; full=np.zeros(len(f['vf'])); full[f['va']]=vr; bound=1+lam
   cs=np.r_[np.where(f['va'],full,bound+loo_cdf(f['vf'])),np.where(f['ua'],ur,bound+reference_cdf(f['vf'],f['uf']))]; current=oscr(isk,correct,cs)
   for beta in BETAS:
    vs,us,_,_=build_retaining_scores(f,lam,beta)
    rows.append({'split':s,'seed':seed,'held_out_emitter':held,'candidate':CANDIDATE,'lambda':lam,'beta':beta,'current_oscr':current,'retaining_oscr':oscr(isk,correct,np.r_[vs,us]),'outer_rogue_accessed':False})
  print(json.dumps({'worker':worker,'completed':[s,seed,held]}),flush=True)
 write_csv(out/f'inner_worker{worker}.csv',rows)

def select(out):
 rows=[]
 for p in sorted(out.glob('inner_worker*.csv')): rows.extend(csv.DictReader(p.open()))
 selected=[]
 for s in SPLITS:
  for seed in SEEDS:
   lc=[]
   for lam in LAMBDAS:
    rr=[r for r in rows if int(r['split'])==s and int(r['seed'])==seed and float(r['lambda'])==lam and float(r['beta'])==0]
    lc.append((np.mean([float(r['current_oscr']) for r in rr]),min(float(r['current_oscr']) for r in rr),-lam,lam))
   lam=max(lc)[3]; bc=[]
   for b in BETAS:
    rr=[r for r in rows if int(r['split'])==s and int(r['seed'])==seed and float(r['lambda'])==lam and float(r['beta'])==b]
    bc.append((np.mean([float(r['retaining_oscr']) for r in rr]),min(float(r['retaining_oscr']) for r in rr),-b,b))
   beta=max(bc)[3]; selected.append({'split':s,'seed':seed,'candidate':CANDIDATE,'lambda':lam,'beta':beta})
 write_csv(out/'selected_lambda_beta.csv',selected); return {(int(r['split']),int(r['seed'])):r for r in selected}

def outer(root,out,batch,device):
 sel=select(out); data=root/'dataset/multiknown17_v1'; y=np.load(data/'Y_global.npy'); role=np.load(data/'sample_role.npy'); rows=[]
 for s in SPLITS:
  m=json.loads((data/f'split_{s}.json').read_text()); known=list(map(int,m['known_classes'])); rogue=list(map(int,m['rogue_classes']))
  for seed in SEEDS:
   r=sel[(s,seed)]; lam=float(r['lambda']); beta=float(r['beta']); tr=np.flatnonzero(np.isin(y,known)&(role==0)); va=np.flatnonzero(np.isin(y,known)&(role==1)); ki=np.flatnonzero(np.isin(y,known)&(role==2)); ui=np.flatnonzero(np.isin(y,rogue)&(role==2)); test=np.r_[ki,ui]
   hp=out/'outer_selected'/f'split{s}_seed{seed}_{CANDIDATE}'/'stage_b/best.pt'
   f=common_features(root,s,seed,known,tr,va,test,root/f'results/multiknown17_auxce_backbone_split{s}_seed{seed}/best.pt',hp,CANDIDATE,batch,device)
   nk=len(ki); isk=np.r_[np.ones(nk,bool),np.zeros(len(ui),bool)]; mapped=np.asarray([known.index(int(v)) for v in y[ki]]); correct=np.r_[f['h_unknown']['logits'][:nk].argmax(1)==mapped,np.zeros(len(ui),bool)]
   fs=f['uf']; fr=fs>f['tau_f']; review=f['unk_sh']+lam*f['unk_sc']; tau=quantile(f['val_sh']+lam*f['val_sc'],.95,'higher'); reject=(~f['ua'])|(f['ua']&(review>tau)); _,ret,_,_=build_retaining_scores(f,lam,beta)
   for method,score,rej in [('primary_fisher',fs,fr),('fisher_retaining_gcr',ret,reject)]: rows.append({'split':s,'seed':seed,'candidate':CANDIDATE,'lambda':lam,'beta':beta,'method':method,**metric_row(isk,correct,score,rej)})
   print(json.dumps({'outer_completed':[s,seed],'lambda':lam,'beta':beta}),flush=True)
 write_csv(out/'complementary_outer_results.csv',rows)

def compare(root,out):
 comp=list(csv.DictReader((out/'complementary_outer_results.csv').open())); full=list(csv.DictReader((root/'results/architecture_simplification_v1/experiment1_fixed_ast/fixed_ast_outer_results.csv').open())); metrics=('auc','eer','oscr','known_acc','known_accept','known_correct_accept','unknown_recall','unknown_f1','overall_open_accuracy'); paired=[]
 for method in ('primary_fisher','fisher_retaining_gcr'):
  a=[r for r in comp if r['method']==method]; b=[r for r in full if r['method']==method]
  for m in metrics:
   av=np.array([float(r[m]) for r in a]); bv=np.array([float(r[m]) for r in b]); d=av-bv; mean,lo,hi=bootstrap_ci(d)
   paired.append({'method':method,'metric':m,'complementary_mean':av.mean(),'current_full_mean':bv.mean(),'mean_difference':mean,'ci95_low':lo,'ci95_high':hi,'plus':int((d>1e-12).sum()),'tie':int((abs(d)<=1e-12).sum()),'minus':int((d<-1e-12).sum()),'min_run_delta':d.min(),'max_run_delta':d.max()})
 write_csv(out/'complementary_vs_current_full_paired.csv',paired)
 p={(r['method'],r['metric']):r for r in paired}; osr=p[('fisher_retaining_gcr','oscr')]; ka=p[('fisher_retaining_gcr','known_acc')]
 if float(osr['ci95_low'])>=-.005 and float(ka['ci95_low'])>=-.005 and float(osr['min_run_delta'])>=-.02: verdict='REMOVE'
 elif float(osr['mean_difference'])>=-.005: verdict='WEAKEN/OPTIONAL'
 else: verdict='KEEP'
 # alpha distribution and implementation cost
 fmap=fixed_map(root); ad={str(a):sum(c.endswith(a) for c in fmap.values()) for a in ('alpha0','alpha025','alpha050','alpha100')}
 stats=[]
 for s in SPLITS:
  for seed in SEEDS:
   fullcand=fmap[(s,seed)]; fp=root/'results/architecture_simplification_v1/experiment1_fixed_ast/outer_selected'/f'split{s}_seed{seed}_{fullcand}'/'stage_b'
   if not fp.exists(): fp=root/'results/inner_physical_arch_selection_v1/outer_selected'/f'split{s}_seed{seed}_{fullcand}'/'stage_b'
   cp=out/'outer_selected'/f'split{s}_seed{seed}_{CANDIDATE}'/'stage_b'
   fc=torch.load(fp/'best.pt',map_location='cpu'); cc=torch.load(cp/'best.pt',map_location='cpu'); ftime=json.loads((fp/'fit_result.json').read_text())['train_seconds']; ctime=json.loads((cp/'fit_result.json').read_text())['train_seconds']
   total=lambda ck:sum(v.numel() for k,v in ck['head'].items() if not k.startswith('view_index_'))
   stats.append({'split':s,'seed':seed,'current_candidate':fullcand,'current_total_head_params':total(fc),'complementary_total_head_params':total(cc),'deployed_params_removed_if_gate_active':2113 if float(fc['residual_alpha'])>0 else 0,'current_train_seconds':ftime,'complementary_train_seconds':ctime,'inference_MACs_removed_if_gate_active':6496 if float(fc['residual_alpha'])>0 else 0})
 write_csv(out/'architecture_costs.csv',stats); (out/'EXPERIMENT2_VERDICT.json').write_text(json.dumps({'verdict':verdict,'rule':'REMOVE if final OSCR and Known Acc lower CI >= -0.5 pp and no run OSCR loss >2 pp; OPTIONAL if mean OSCR loss <=0.5 pp; else KEEP','alpha_selection_distribution':ad},indent=2)+'\n')

def main():
 p=argparse.ArgumentParser(); p.add_argument('mode',choices=('inner','outer','compare')); p.add_argument('--root',default=str(Path(__file__).resolve().parents[1])); p.add_argument('--out',default='results/architecture_simplification_v1/experiment2_complementary_only'); p.add_argument('--worker',type=int,default=0); p.add_argument('--workers',type=int,default=1); p.add_argument('--device',default='cuda:0'); p.add_argument('--batch-size',type=int,default=128); a=p.parse_args(); root=Path(a.root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); dev=torch.device(a.device if torch.cuda.is_available() else 'cpu')
 if a.mode=='inner': inner(root,out,a.batch_size,dev,a.worker,a.workers)
 elif a.mode=='outer': outer(root,out,a.batch_size,dev)
 else: compare(root,out)
if __name__=='__main__': main()
