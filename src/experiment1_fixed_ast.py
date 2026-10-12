#!/usr/bin/env python3
"""Experiment 1: current selective architecture versus protocol-legal Fixed-AST."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np, torch
from calibrated_rejection_fusion_multiknown import loo_cdf, reference_cdf
from continuous_gcr_loeo import quantile
from evaluate_new_arch_gcr_matched import common_features, metric_row, oscr, selected_map, write_csv
from fisher_retaining_cascade import BETAS, build_retaining_scores, bootstrap_ci, lambda_map

SPLITS=(1,2,3); SEEDS=(42,2023,3407); LAMBDAS=(0.,.25,.5,1.,2.,4.)

def fixed_map(root):
    rows=list(csv.DictReader((root/'results/inner_physical_arch_selection_v1/inner_arch_selection_summary.csv').open()))
    out={}
    amap={'0':0.,'025':.25,'050':.5,'100':1.}
    for s in SPLITS:
      for seed in SEEDS:
        rr=[r for r in rows if int(r['split'])==s and int(r['seed'])==seed and r['candidate'].startswith('nophase')]
        key=lambda r:(float(r['mean_inner_oscr']),float(r['worst_fold_oscr']),float(r['mean_known_acc']),-amap[r['candidate'].split('alpha')[1]])
        out[(s,seed)]=max(rr,key=key)['candidate']
    return out

def beta_map(path):
    out={}
    for r in csv.DictReader(path.open()):
      if r['selected'].lower()=='true': out[(int(r['split']),int(r['seed']))]=float(r['beta'])
    return out

def fold_features(root,s,seed,held,candidate,batch,device):
    data=root/'dataset/multiknown17_v1'; y=np.load(data/'Y_global.npy'); role=np.load(data/'sample_role.npy')
    known=list(map(int,json.loads((data/f'split_{s}.json').read_text())['known_classes'])); active=[v for v in known if v!=held]
    tr=np.flatnonzero(np.isin(y,active)&(role==0)); va=np.flatnonzero(np.isin(y,active)&(role==1)); pu=np.flatnonzero(y==held)
    tag=f'split{s}_seed{seed}_heldout{held}'
    f=common_features(root,s,seed,active,tr,va,pu,root/'results/continuous_gcr_loeo_v2/inner'/tag/'stage_a/best.pt',root/'results/inner_physical_arch_selection_v1/inner'/tag/candidate/'stage_b/best.pt',candidate,batch,device)
    isk=np.r_[np.ones(len(va),bool),np.zeros(len(pu),bool)]; correct=np.r_[f['h_val']['logits'].argmax(1)==f['val_y'],np.zeros(len(pu),bool)]
    return f,isk,correct

def inner(root,out,batch,device):
    cur=selected_map(root); fixed=fixed_map(root); rows=[]
    changed=[k for k in fixed if fixed[k]!=cur[k]]
    for s,seed in changed:
      known=list(map(int,json.loads((root/'dataset/multiknown17_v1'/f'split_{s}.json').read_text())['known_classes']))
      for held in known:
        f,isk,correct=fold_features(root,s,seed,held,fixed[(s,seed)],batch,device)
        for lam in LAMBDAS:
          vr=f['val_sh']+lam*f['val_sc']; ur=f['unk_sh']+lam*f['unk_sc']; bound=1+lam
          vr_full=np.zeros(len(f['vf']),float); vr_full[f['va']]=vr
          cs=np.r_[np.where(f['va'],vr_full,bound+loo_cdf(f['vf'])),np.where(f['ua'],ur,bound+reference_cdf(f['vf'],f['uf']))]
          current=oscr(isk,correct,cs)
          for beta in BETAS:
            vs,us,_,_=build_retaining_scores(f,lam,beta)
            rows.append({'split':s,'seed':seed,'held_out_emitter':held,'candidate':fixed[(s,seed)],'lambda':lam,'beta':beta,'current_oscr':current,'retaining_oscr':oscr(isk,correct,np.r_[vs,us]),'outer_rogue_accessed':False})
        print(json.dumps({'completed':[s,seed,held]}),flush=True)
    write_csv(out/'changed_inner_all.csv',rows)

def select_review(root,out):
    current_l=lambda_map(root/'results/new_arch_gcr_matched_v1'); current_b=beta_map(root/'results/fisher_retaining_cascade_v1/beta_selection_summary.csv')
    cur=selected_map(root); fixed=fixed_map(root); rows=list(csv.DictReader((out/'changed_inner_all.csv').open())); selected=[]
    for key,candidate in fixed.items():
      s,seed=key
      if candidate==cur[key]: lam,beta=current_l[key],current_b[key]; source='reused_identical_architecture'
      else:
        cand=[]
        for lam0 in LAMBDAS:
          rr=[r for r in rows if int(r['split'])==s and int(r['seed'])==seed and float(r['lambda'])==lam0 and float(r['beta'])==0]
          cand.append((np.mean([float(r['current_oscr']) for r in rr]),min(float(r['current_oscr']) for r in rr),-lam0,lam0))
        lam=max(cand)[3]
        cand=[]
        for b in BETAS:
          rr=[r for r in rows if int(r['split'])==s and int(r['seed'])==seed and float(r['lambda'])==lam and float(r['beta'])==b]
          cand.append((np.mean([float(r['retaining_oscr']) for r in rr]),min(float(r['retaining_oscr']) for r in rr),-b,b))
        beta=max(cand)[3]; source='reselected_strict_inner_loeo'
      selected.append({'split':s,'seed':seed,'candidate':candidate,'lambda':lam,'beta':beta,'source':source})
    write_csv(out/'fixed_ast_selected.csv',selected); return {(r['split'],r['seed']):r for r in selected}

def outer_head(root,out,s,seed,candidate):
    p=out/'outer_selected'/f'split{s}_seed{seed}_{candidate}'/'stage_b/best.pt'
    if p.exists(): return p
    return root/'results/inner_physical_arch_selection_v1/outer_selected'/f'split{s}_seed{seed}_{candidate}'/'stage_b/best.pt'

def outer(root,out,batch,device):
    sel=select_review(root,out); data=root/'dataset/multiknown17_v1'; y=np.load(data/'Y_global.npy'); role=np.load(data/'sample_role.npy'); rows=[]
    for s in SPLITS:
      m=json.loads((data/f'split_{s}.json').read_text()); known=list(map(int,m['known_classes'])); rogue=list(map(int,m['rogue_classes']))
      for seed in SEEDS:
        r=sel[(s,seed)]; candidate=r['candidate']; lam=float(r['lambda']); beta=float(r['beta'])
        tr=np.flatnonzero(np.isin(y,known)&(role==0)); va=np.flatnonzero(np.isin(y,known)&(role==1)); ki=np.flatnonzero(np.isin(y,known)&(role==2)); ui=np.flatnonzero(np.isin(y,rogue)&(role==2)); test=np.r_[ki,ui]
        f=common_features(root,s,seed,known,tr,va,test,root/f'results/multiknown17_auxce_backbone_split{s}_seed{seed}/best.pt',outer_head(root,out,s,seed,candidate),candidate,batch,device)
        nk=len(ki); isk=np.r_[np.ones(nk,bool),np.zeros(len(ui),bool)]; mapped=np.asarray([known.index(int(v)) for v in y[ki]]); correct=np.r_[f['h_unknown']['logits'][:nk].argmax(1)==mapped,np.zeros(len(ui),bool)]
        fs=f['uf']; fr=fs>f['tau_f']; review=f['unk_sh']+lam*f['unk_sc']; tau=quantile(f['val_sh']+lam*f['val_sc'],.95,'higher'); reject=(~f['ua'])|(f['ua']&(review>tau)); _,ret,_,_=build_retaining_scores(f,lam,beta)
        for method,score,rej in [('primary_fisher',fs,fr),('fisher_retaining_gcr',ret,reject)]:
          rows.append({'split':s,'seed':seed,'candidate':candidate,'lambda':lam,'beta':beta,'method':method,**metric_row(isk,correct,score,rej)})
        print(json.dumps({'outer_completed':[s,seed], 'candidate':candidate}),flush=True)
    write_csv(out/'fixed_ast_outer_results.csv',rows)

def compare(root,out):
    fixed=list(csv.DictReader((out/'fixed_ast_outer_results.csv').open())); current=list(csv.DictReader((root/'results/fisher_retaining_cascade_v1/outer_fisher_retaining_results.csv').open()))
    cmap={'primary_fisher':'primary_fisher','fisher_retaining_gcr':'fisher_retaining_review'}; metrics=('auc','eer','oscr','known_acc','known_accept','known_correct_accept','unknown_recall','unknown_f1','overall_open_accuracy'); summary=[]; paired=[]
    for fm,cm in cmap.items():
      fr=[r for r in fixed if r['method']==fm]; cr=[r for r in current if r['method']==cm]
      for metric in metrics:
        fv=np.array([float(r[metric]) for r in fr]); cv=np.array([float(r[metric]) for r in cr]); d=fv-cv; mean,lo,hi=bootstrap_ci(d)
        paired.append({'method':fm,'metric':metric,'fixed_mean':fv.mean(),'current_mean':cv.mean(),'mean_difference':mean,'ci95_low':lo,'ci95_high':hi,'plus':int((d>1e-12).sum()),'tie':int((abs(d)<=1e-12).sum()),'minus':int((d<-1e-12).sum())})
    write_csv(out/'fixed_vs_current_paired.csv',paired)
    # Pre-registered simplification decision: non-inferior final OSCR and Known Acc within 0.5 pp, no >2 pp run loss.
    p={(r['method'],r['metric']):r for r in paired}; ok=float(p[('fisher_retaining_gcr','oscr')]['ci95_low'])>=-.005 and float(p[('fisher_retaining_gcr','known_acc')]['ci95_low'])>=-.005
    fr=[r for r in fixed if r['method']=='fisher_retaining_gcr']; cr=[r for r in current if r['method']=='fisher_retaining_review']; losses=[float(a['oscr'])-float(b['oscr']) for a,b in zip(fr,cr)]; ok=ok and min(losses)>=-.02
    (out/'EXPERIMENT1_DECISION.json').write_text(json.dumps({'fixed_ast_supported':bool(ok),'rule':'lower paired CI for final OSCR and Known Acc >= -0.5 pp; no run OSCR loss >2 pp','min_run_oscr_delta':min(losses)},indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('inner','outer','compare')); p.add_argument('--root',default=str(Path(__file__).resolve().parents[1])); p.add_argument('--out',default='results/architecture_simplification_v1/experiment1_fixed_ast'); p.add_argument('--device',default='cuda:0'); p.add_argument('--batch-size',type=int,default=128); a=p.parse_args(); root=Path(a.root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); dev=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    if a.mode=='inner': inner(root,out,a.batch_size,dev)
    elif a.mode=='outer': outer(root,out,a.batch_size,dev)
    else: compare(root,out)
if __name__=='__main__': main()
