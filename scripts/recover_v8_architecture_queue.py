"""Preserve failure evidence, resume current DenseNet, then finalize its queue."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import psutil
import torch
import train_v8_architecture as trainer
import resilient_training_io as io
from run_v8_architecture_queue import verified_result

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    report=Path(args.report).resolve()
    if not report.is_relative_to((ROOT/'reports/v8_architectures').resolve()):
        raise ValueError('Expected architecture report directory')
    queue=json.loads((report/'queue.json').read_text(encoding='utf-8'))
    if queue['status']!='failed' or queue['current_model']!='densenet121':
        raise ValueError('Expected the interrupted DenseNet queue')
    active=json.loads((ROOT/'reports/_qa/v8_staged_active.json').read_text(encoding='utf-8'))
    run=Path(active['run']).resolve()
    state=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    identity=state['identity']
    if (state['model_name']!='densenet121' or state['complete']
            or identity['source_hashes']!=trainer.fingerprints()
            or identity['dataset']['dataset_json_sha256']!=queue['dataset_sha256']
            or identity['cache_manifest_sha256']!=queue['cache_manifest_sha256']):
        raise ValueError('Checkpoint mismatch; recovery refused')
    meta=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    data,cache=Path(meta['data']).resolve(),Path(meta['cache']).resolve()
    for row in queue['completed']:
        if row!=verified_result(row['run'],row['model'],data,cache,42,row['reused']):
            raise ValueError('Completed result changed')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    archive=report/('io_recovery_'+stamp)
    archive.mkdir()
    for name in ('queue.json','validation_summary.json'):
        shutil.copy2(report/name,archive/name)
    for name in ('last.pt','best.pt','run_config.json','history.json','continue_state.json','live_progress.json','live_progress.json.tmp'):
        if (run/name).exists():
            shutil.copy2(run/name,archive/('densenet_'+name))
    for name in ('resilient_training_io.py','resume_v8_with_resilient_io.py','recover_v8_architecture_queue.py'):
        shutil.copy2(ROOT/'scripts'/name,archive/name)
    recovery=dict(archived_failure=str(archive),completed_epoch=state['epoch'],
        resume_checkpoint=str(run/'last.pt'),previous_error=queue.get('error'),
        reason='PermissionError updating telemetry; bounded atomic write retries and best-effort telemetry',
        created=datetime.now().astimezone().isoformat())
    queue.setdefault('recovery_attempts',[]).append(recovery)
    queue.pop('error',None)
    queue.update(status='running',pid=os.getpid(),process_created=psutil.Process().create_time())
    def save():
        io.write_json(report/'queue.json',queue)
        io.write_json(report/'validation_summary.json',queue['completed'])
    save()
    del state
    try:
        subprocess.run([sys.executable,'-u','-B',str(ROOT/'scripts/resume_v8_with_resilient_io.py'),
            '--model','densenet121','--data',str(data),'--cache-dir',str(cache),'--seed','42',
            '--resume',str(run/'last.pt')],check=True,cwd=ROOT)
        row=verified_result(run,'densenet121',data,cache,42)
        target=report/'evidence/densenet121';target.mkdir(exist_ok=False)
        for name in ('run_config.json','history.json','continue_state.json','io_adapter.json'):
            shutil.copy2(run/name,target/name)
        queue['completed'].append(row)
        queue.update(status='complete',current_model=None);save()
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/report_v8_architecture_queue.py'),
            '--report',str(report)],check=True,cwd=ROOT)
    except Exception as error:
        queue.update(status='failed',error=str(error));save()
        raise


if __name__=='__main__':
    main()
