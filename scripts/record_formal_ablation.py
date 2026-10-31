"""Validate and snapshot the completed cache-v1 formal batch; never trains."""
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'experiments/oral_v7_resnet18_baseline_20261026/code'
BATCH = ROOT / 'reports/ablation/20261029_141726'
VARIANTS = [('resnet50', 'none', 'ce'), ('efficientnet_b3', 'none', 'ce'),
            ('densenet121', 'none', 'ce'), ('resnet50', 'se', 'ce'),
            ('resnet50', 'cbam', 'ce'), ('resnet50', 'none', 'weighted_ce'),
            ('resnet50', 'none', 'label_smoothing'), ('resnet50', 'none', 'focal')]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record():
    rows = read(BATCH / 'validation_summary.json')
    require(sorted(r['variant'] for r in rows) == list(range(8)), 'Expected exactly eight variants')
    verified = []
    for row in sorted(rows, key=lambda r: r['variant']):
        index = row['variant']
        run = (ROOT / row['run']).resolve()
        require(run.is_relative_to(ROOT), 'Run escapes workspace')
        meta, history, state = [read(run / name) for name in
                                ('run_config.json', 'history.json', 'continue_state.json')]
        cfg = meta['config']
        require(tuple(row[k] for k in ('model', 'attention', 'loss')) == VARIANTS[index], 'Variant mismatch')
        require(tuple(cfg[k] for k in ('model', 'attention', 'loss')) == VARIANTS[index], 'Config mismatch')
        require(not row['pilot'] and row['seed'] == 42 and row['imgsz'] == 640, 'Not formal seed42/640')
        expected = dict(seed=42, imgsz=640, epochs=100, patience=20, batch=4,
                        workers=0, lr0=.001, optimizer='AdamW', pretrained=True)
        require(all(cfg.get(k) == v for k, v in expected.items()), 'Common configuration differs')
        require([h['epoch'] for h in history] == list(range(1, len(history) + 1)), 'Incomplete history')
        require(state['status'] == 'complete' and state['last_completed_epoch'] == len(history), 'Run incomplete')
        selected = min(history, key=lambda h: h['val_metrics']['loss'])
        require(selected['epoch'] == row['best_epoch'] == state['best_epoch'], 'Best epoch differs')
        require(selected['val_metrics'] == row['validation'], 'Metrics differ from history')
        require(state['best_val_loss'] == selected['val_metrics']['loss'], 'State CE differs')
        require(all(h['train_images'] == 893 and h['val_metrics']['images'] == 112
                    and h['val_metrics']['threshold'] == .5 for h in history), 'Evaluation universe differs')
        require(meta['dataset']['dataset_json_sha256'] == state['dataset_json_sha256'], 'Dataset state differs')
        require(len(history) == selected['epoch'] + 20, 'Unexpected stopping boundary')
        verified.append((row, run, meta, selected, len(history)))
    require(len({v[2]['dataset']['dataset_json_sha256'] for v in verified}) == 1, 'Datasets differ')
    require(len({json.dumps(v[2]['preprocessing'], sort_keys=True) for v in verified}) == 1, 'Preprocessing differs')
    require(len({json.dumps(v[2]['augmentation'], sort_keys=True) for v in verified}) == 1, 'Augmentations differ')
    require(len({json.dumps(v[2]['source_hashes'], sort_keys=True) for v in verified}) == 1, 'Training source differs')
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    destination = ROOT / 'reports/ablation/records' / ('formal_' + now.strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    for row, run, meta, selected, epochs in verified:
        artifacts = {}
        for name in ('run_config.json', 'history.json', 'continue_state.json',
                     'training_config.yaml', 'best.pt', 'last.pt'):
            path = run / name
            artifacts[name] = dict(path=path.relative_to(ROOT).as_posix(),
                                   bytes=path.stat().st_size, sha256=digest(path))
            if path.suffix != '.pt':
                target = destination / 'evidence' / f"variant_{row['variant']}" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                require(digest(target) == artifacts[name]['sha256'], 'Snapshot readback failed')
        results.append(dict(**row, completed_epochs=epochs, stop_reason='patience_20',
                            selected_train_loss=selected['train_loss'], metadata=meta, artifacts=artifacts))
    for path in BATCH.iterdir():
        if path.suffix in ('.json', '.yaml'):
            target = destination / 'evidence' / 'batch' / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    source_checks = {}
    for name, expected_hash in verified[0][2]['source_hashes'].items():
        path = CODE / 'src/oral_fluorescence_detection' / name
        actual = digest(path)
        source_checks[name] = dict(recorded=expected_hash, current=actual, matches=actual == expected_hash)
        target = destination / 'evidence' / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    payload = dict(recorded_at=now.isoformat(), status='8_of_8_complete', results=results,
                   source_checks=source_checks, selection='minimum unweighted unsmoothed whole-validation CE',
                   threshold=.5, test_performance_computed=False,
                   archive_scope='Metadata/history/config snapshots; weight paths and hashes, not weight copies.',
                   limitations=['One seed; exploratory selection on 112 validation images.',
                                'Candidate labels, not confirmed clinical diagnoses.',
                                'Source groups are not confirmed independent patients.',
                                'Previously evaluated test set is not a new blind holdout.'])
    (destination / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['variant', 'model', 'attention', 'loss', 'seed', 'completed_epochs', 'best_epoch',
              'val_ce', 'auroc', 'balanced_accuracy', 'sensitivity', 'specificity', 'accuracy']
    with (destination / 'results.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            values = {k: row[k] for k in fields[:7]}
            values.update(val_ce=row['validation']['loss'])
            values.update({k: row['validation'][k] for k in fields[8:]})
            writer.writerow(values)
    lines = ['# 八组缓存正式实验结果', '', f'记录时间：{now.isoformat()}', '',
             '8/8组完成，均在最佳轮后20轮早停。第0组复用同配置已完成基线，来源保存在results.json。',
             '共同设置：640、batch4、seed42、AdamW lr0.001、最多100轮；893张训练、112张验证。',
             '按整体验证集无权重无平滑CE选模，固定阈值0.5。本次没有计算测试集性能。', '',
             '| 编号 | 模型 | 注意力 | 损失 | 完成/最佳轮 | 验证CE | AUROC | 平衡准确率 | 敏感度 | 特异度 |',
             '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in results:
        v = row['validation']
        lines.append(f"| {row['variant']} | {row['model']} | {row['attention']} | {row['loss']} | "
                     f"{row['completed_epochs']}/{row['best_epoch']} | " + ' | '.join(
                         f'{v[k]:.4f}' for k in ('loss', 'auroc', 'balanced_accuracy', 'sensitivity', 'specificity')) + ' |')
    lines += ['', '最低CE候选为ResNet50+SE+CE：CE0.3362、AUROC0.9447、平衡准确率0.8694。',
              'weighted CE按规则选中的第1轮将112张全部预测为阳性候选（TN0、FP43、FN0、TP69）；保留失败记录，不能据此称所有轮都如此。',
              '单种子比较不足以证明稳定改进；多随机种子仅衡量训练波动，不增加独立患者或验证样本数。',
              '标签为候选标签，来源组不等于独立患者；不宣称临床诊断效能。', '',
              '证据：results.json保留完整指标、配置、环境版本、预处理/数据/源码指纹及best/last权重SHA-256。',
              'evidence保存配置、逐轮历史、完成状态和当前源码快照；source_checks标明源码是否匹配训练记录。',
              '权重保留原实验目录，仅记录路径与哈希；此目录不是含权重和数据的独立迁移交付包。']
    (destination / '结果报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    inventory = {p.relative_to(destination).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(destination.rglob('*')) if p.is_file()}
    (destination / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    require(all(digest(destination / p) == v['sha256'] for p, v in inventory.items()), 'Inventory readback failed')
    print(json.dumps(dict(record=str(destination), variants=len(results), source_checks=source_checks), ensure_ascii=False))


if __name__ == '__main__':
    record()
