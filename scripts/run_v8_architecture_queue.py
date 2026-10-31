"""Reuse verified v8 ResNet baseline, then train fresh EfficientNet/DenseNet serially."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import psutil

import train_v8_staged_baseline as baseline
from oral_fluorescence_detection.image_cache import sha256
from oral_fluorescence_detection.image_classification import _write_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/v8_architectures'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verified_result(run, model, data, cache, seed, original_baseline=False):
    run = Path(run).resolve()
    meta, state, history = [read(run/name) for name in ('run_config.json','continue_state.json','history.json')]
    identity = meta['identity']
    cfg = identity['protocol']
    fixed = {key:value for key,value in baseline.PROTOCOL.items()
             if key not in ('version','model','initialization','stage2_trainable')}
    if (state['status']!='complete' or not history or identity['seed']!=seed or cfg['model']!=model
            or any(cfg.get(k)!=v for k,v in fixed.items())
            or identity['dataset']['dataset_json_sha256']!=sha256(data)
            or identity['cache_manifest_sha256']!=sha256(cache/'manifest.json')
            or Path(meta['data']).resolve()!=data or Path(meta['cache']).resolve()!=cache):
        raise ValueError('Run is incomplete or incompatible with the controlled v8 comparison')
    if original_baseline and (cfg!=baseline.PROTOCOL or identity['source_hashes']!=baseline.fingerprints()):
        raise ValueError('Baseline protocol/source differs from recorded formal run')
    for i,row in enumerate(history,1):
        stage=baseline.stage_for_epoch(i)
        if (row['epoch']!=i or row['stage']!=stage or row['lr']!=cfg['learning_rates'][stage-1]
                or row['train_images']!=782 or row['val_metrics']['images']!=223):
            raise ValueError('Training history does not implement the frozen v8 stage schedule')
    if len(history)<71 or history[-1]['stage3_stale']!=10:
        raise ValueError('Expected stage3 convergence stop')
    best=min(history,key=lambda row:row['val_metrics']['loss'])
    if state['best_epoch']!=best['epoch']:
        raise ValueError('Best checkpoint epoch differs from validation selection')
    return dict(model=model, run=str(run), reused=original_baseline, epochs=len(history),
        best_epoch=best['epoch'], validation=best['val_metrics'], seed=seed,
        weight_sha256={name:sha256(run/name) for name in ('best.pt','last.pt')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    data,cache=Path(args.data).resolve(),Path(args.cache_dir).resolve()
    result=verified_result(args.baseline,'resnet50',data,cache,args.seed,True)
    common=['--data',str(data),'--cache-dir',str(cache),'--seed',str(args.seed)]
    trainer=ROOT/'scripts/train_v8_architecture.py'
    if args.check_only:
        for model in ('efficientnet_b3','densenet121'):
            subprocess.run([sys.executable,'-u','-B',str(trainer),'--model',model,*common,'--check-only'],check=True,cwd=ROOT)
        print('QUEUE PREFLIGHT PASSED; baseline reusable; no training executed',flush=True)
        return
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    report=REPORTS/stamp
    report.mkdir(parents=True,exist_ok=False)
    evidence=report/'evidence';evidence.mkdir()
    for source in (Path(__file__),trainer,ROOT/'scripts/v8_architecture_support.py',Path(baseline.__file__)):
        shutil.copy2(source,evidence/source.name)
    for name in ('run_config.json','history.json','continue_state.json'):
        shutil.copy2(Path(args.baseline)/name,evidence/('baseline_'+name))
    queue=dict(scope='three architectures; seed42; no attention/loss ablations; no test evaluation',
        status='running', models=['resnet50','efficientnet_b3','densenet121'],
        current_model=None, completed=[result], report=str(report),
        pid=os.getpid(), process_created=psutil.Process().create_time(),
        dataset_sha256=sha256(data), cache_manifest_sha256=sha256(cache/'manifest.json'),
        created=datetime.now().astimezone().isoformat())
    pointer=ROOT/'reports/_qa/v8_architecture_queue.json'
    def save():
        _write_json(report/'queue.json',queue)
        _write_json(pointer,dict(report=str(report)))
        _write_json(report/'validation_summary.json',queue['completed'])
    save()
    print(f'ARCHITECTURE_REPORT={report}',flush=True)
    try:
        for model in ('efficientnet_b3','densenet121'):
            queue['current_model']=model;save()
            print(f'Starting fresh ImageNet architecture: {model}',flush=True)
            subprocess.run([sys.executable,'-u','-B',str(trainer),'--model',model,*common],check=True,cwd=ROOT)
            active=read(ROOT/'reports/_qa/v8_staged_active.json')
            row=verified_result(active['run'],model,data,cache,args.seed)
            target=evidence/model;target.mkdir()
            for name in ('run_config.json','history.json','continue_state.json'):
                shutil.copy2(Path(active['run'])/name,target/name)
            queue['completed'].append(row);save()
        queue['status']='complete';queue['current_model']=None;save()
        print(f'ARCHITECTURE COMPARISON COMPLETE: {report}; validation only',flush=True)
    except Exception as error:
        queue['status']='failed';queue['error']=str(error);save()
        raise


if __name__=='__main__':
    main()
