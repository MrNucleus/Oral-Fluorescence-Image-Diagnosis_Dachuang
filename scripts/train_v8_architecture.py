"""Section 4.3 architecture comparison; isolated from the completed baseline source.

Train and validation only. Fresh runs always start from generic ImageNet weights.
Resume is restricted to this protocol, source version and frozen dataset.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments/oral_v7_resnet18_baseline_20261026/code"
sys.path.insert(0, str(CODE / "src"))
from oral_fluorescence_detection.image_classification import (
    build_model, evaluate_model, seed_worker, _atomic_torch_save, _write_json,
    _capture_rng_state, _restore_rng_state, _new_run,
)
from oral_fluorescence_detection.online_augmentation import (
    validate_handoff, AugmentationPolicy, CLASS_NAMES,
)
from oral_fluorescence_detection.image_cache import CachedImageClassificationDataset, sha256

from v8_architecture_support import WEIGHTS, LAST_STAGES, build_classifier, configure_stage as apply_architecture_stage

PROTOCOL = dict(version="v8-section4.3-architectures-v1", model="resnet50", attention="none",
    stage1_epochs=20, stage2_epochs=40, learning_rates=[1e-3, 1e-4, 1e-5],
    optimizer="Adam", weight_decay=1e-4, dropout=0.5, label_smoothing=0.1,
    loss="class_weighted_smoothed_ce", class_weight_formula="N/(2*n_class)",
    patience=10, early_stopping_scope="stage3; reset on entry; fixed stages1/2",
    stage3_max_epochs=None, imgsz=640, batch=4, workers=0, precision="FP32",
    checkpoint_selection="minimum overall validation ordinary unweighted CE",
    threshold=0.5, augmentation="geometry; whole-image resize/pad; no crops",
    initialization="ResNet50_Weights.IMAGENET1K_V2; new head random",
    stage2_trainable="layer3,layer4,fc; other parameters and BN buffers frozen",
    optimizer_transition="retain Adam state; set stage learning rate")


def stage_for_epoch(epoch):
    if epoch < 1:
        raise ValueError("Epochs are one based")
    return 1 if epoch <= 20 else 2 if epoch <= 60 else 3


def configure_stage(model, optimizer, stage):
    return apply_architecture_stage(model, optimizer, PROTOCOL['model'], stage)


def make_model(pretrained):
    return build_classifier(PROTOCOL['model'], pretrained)


def make_loss(labels, device):
    counts = torch.bincount(torch.tensor(labels), minlength=2)
    if counts.numel() != 2 or (counts == 0).any():
        raise ValueError("Both training classes required")
    weights = len(labels) / (2 * counts.float())
    return nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1), {
        'counts': counts.tolist(), 'weights': weights.tolist(), 'smoothing': 0.1}


def fingerprints():
    paths = [Path(__file__), Path(__file__).with_name('v8_architecture_support.py'), *[CODE / 'src/oral_fluorescence_detection' / name
        for name in ('image_classification.py', 'online_augmentation.py', 'image_cache.py', 'ablation.py')]]
    return {p.name: sha256(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=tuple(WEIGHTS))
    parser.add_argument('--data', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', help='Only last.pt from this exact v8 protocol')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    PROTOCOL.update(model=args.model, initialization=str(WEIGHTS[args.model])+'; new head random',
        stage2_trainable=list(LAST_STAGES[args.model])+['fc' if args.model=='resnet50' else 'classifier'])
    torch.set_num_threads(4)
    data, cache = Path(args.data).resolve(), Path(args.cache_dir).resolve()
    summary = validate_handoff(data)
    if summary['images'] != 1117 or [summary['splits'][s]['images'] for s in ('train','val','test')] != [782,223,112]:
        raise ValueError('Expected frozen v8 782/223/112 cohort')
    if summary['dataset_json_sha256'] != 'ad821b42506c0203dcbaf6b646095c187470f9cd813feca65a115c0a12ae16be':
        raise ValueError('Expected exact frozen v8 manifest')
    policy = AugmentationPolicy(profile='geometry')
    datasets = {s: CachedImageClassificationDataset(data, s, policy, 640, cache_dir=cache)
                for s in ('train','val')}
    identity = dict(protocol=PROTOCOL, seed=args.seed, dataset=summary,
        cache_manifest_sha256=sha256(cache/'manifest.json'), source_hashes=fingerprints())
    _, loss_meta = make_loss([r['label'] for r in datasets['train'].records], 'cpu')
    print(json.dumps(dict(preflight='passed', identity=identity, loss=loss_meta,
                         training_executed=False), ensure_ascii=False), flush=True)
    if args.check_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this formal run')
    device = torch.device('cuda:0')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generators = {s: torch.Generator().manual_seed(args.seed) for s in datasets}
    loaders = {s: DataLoader(ds, batch_size=4, shuffle=s=='train', num_workers=0,
        worker_init_fn=seed_worker, generator=generators[s]) for s, ds in datasets.items()}
    model = make_model(not bool(args.resume)).to(device)
    criterion, loss_meta = make_loss([r['label'] for r in datasets['train'].records], device)
    # Include all parameters from the outset so newly unfrozen parameters can learn.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history, epoch, best, phase3_best, stale = [], 0, float('inf'), float('inf'), 0
    if args.resume:
        resume = Path(args.resume).resolve()
        if resume.name != 'last.pt':
            raise ValueError('Exact continuation requires this run\'s last.pt')
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state.get('identity') != identity:
            raise ValueError('Resume protocol/data/cache/source mismatch; old weights forbidden')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        history, epoch = state['history'], state['epoch']
        best, phase3_best, stale = state['best_loss'], state['phase3_best'], state['stale']
        _restore_rng_state(torch, generators['train'], state['rng_state'])
        run = resume.parent
        if state['complete']:
            print('Run already complete; no further training', flush=True)
            return
    else:
        run = _new_run(CODE/'outputs/image_classifier_runs',
            'v8_721_%s_staged_seed%d_%s' % (args.model, args.seed, datetime.now().strftime('%Y%m%d_%H%M%S')))
        _write_json(run/'run_config.json', dict(identity=identity, data=str(data), cache=str(cache),
            loss=loss_meta, torch=torch.__version__, device=torch.cuda.get_device_name(0),
            created=datetime.now().astimezone().isoformat(), initialization='generic ImageNet; no project checkpoint'))
    print(f'RUN_DIRECTORY={run}', flush=True)
    import psutil
    _write_json(ROOT/'reports/_qa/v8_staged_active.json', dict(run=str(run), data=str(data), cache=str(cache),
        pid=os.getpid(), process_created=psutil.Process().create_time()))
    while True:
        epoch += 1
        stage = stage_for_epoch(epoch)
        trainable = configure_stage(model, optimizer, stage)
        started, total, seen = time.perf_counter(), 0., 0
        def progress(batch=0, phase='train'):
            _write_json(run/'live_progress.json', dict(epoch=epoch, stage=stage,
                lr=optimizer.param_groups[0]['lr'], trainable_parameters=trainable,
                batch=batch, batches=len(loaders['train']), train_images=seen,
                train_loss=total/seen if seen else None, phase=phase,
                updated=datetime.now().astimezone().isoformat()))
        progress()
        print(json.dumps(dict(epoch=epoch, stage=stage, lr=optimizer.param_groups[0]['lr'],
                              trainable_parameters=trainable)), flush=True)
        for i, batch in enumerate(loaders['train'], 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['image'].to(device))
            loss = criterion(logits, batch['label'].to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
            n = len(batch['label'])
            total += float(loss.detach())*n
            seen += n
            if i == 1 or i % 25 == 0 or i == len(loaders['train']):
                progress(i)
                print(json.dumps(dict(epoch=epoch, stage=stage, batch=i,
                    batches=len(loaders['train']), train_images=seen, train_loss=total/seen)), flush=True)
        progress(len(loaders['train']), 'validation')
        metrics, _ = evaluate_model(model, loaders['val'], device)
        if stage == 3:
            if metrics['loss'] < phase3_best:
                phase3_best, stale = metrics['loss'], 0
            else:
                stale += 1
        complete = stage == 3 and stale >= 10
        history.append(dict(epoch=epoch, stage=stage, lr=optimizer.param_groups[0]['lr'],
            trainable_parameters=trainable, train_images=seen, train_loss=total/seen,
            val_metrics=metrics, seconds=time.perf_counter()-started, stage3_stale=stale))
        print(json.dumps(history[-1]), flush=True)
        checkpoint = dict(task='image_classification', model_name=args.model, attention='none',
            dropout=0.5, class_names=CLASS_NAMES, imgsz=640, seed=args.seed,
            modalities=['fluorescence','standard'], augmentation=policy.__dict__,
            preprocessing=dict(version='resize-before-augmentation-v1', cache_manifest_sha256=identity['cache_manifest_sha256']),
            dataset_json_sha256=summary['dataset_json_sha256'], identity=identity,
            model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            epoch=epoch, stage=stage, val_metrics=metrics, loss=loss_meta)
        if metrics['loss'] < best:
            best = metrics['loss']
            _atomic_torch_save(checkpoint, run/'best.pt')
        _write_json(run/'history.json', history)
        _atomic_torch_save(dict(checkpoint, optimizer_state_dict=optimizer.state_dict(), history=history,
            best_loss=best, phase3_best=phase3_best, stale=stale, complete=complete,
            rng_state=_capture_rng_state(torch, generators['train'])), run/'last.pt')
        _write_json(run/'continue_state.json', dict(status='complete' if complete else 'running',
            last_completed_epoch=epoch, stage=stage, best_epoch=min(history,key=lambda r:r['val_metrics']['loss'])['epoch'],
            stage3_stale=stale, resume_checkpoint=str(run/'last.pt')))
        progress(len(loaders['train']), 'complete' if complete else 'epoch_saved')
        if complete:
            print(f'COMPLETE: {run}/best.pt; no test evaluation', flush=True)
            break


if __name__ == '__main__':
    main()
