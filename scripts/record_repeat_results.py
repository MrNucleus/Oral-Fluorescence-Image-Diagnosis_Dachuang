"""Archive completed seed42/43/44 baseline-SE comparisons and a factual report."""
import json
import shutil
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from record_formal_ablation import ROOT, CODE, digest, read, require


def main():
    sources = [ROOT / 'reports/ablation' / stamp / 'validation_summary.json' for stamp in
               ('20261029_141726', '20261030_174612', '20261030_193451')]
    records = []
    for source in sources:
        for row in read(source):
            if row['variant'] not in (0, 3):
                continue
            run = (ROOT / row['run']).resolve()
            require(run.is_relative_to(ROOT), 'Run outside workspace')
            meta, history, state = [read(run / n) for n in
                                    ('run_config.json', 'history.json', 'continue_state.json')]
            selected = min(history, key=lambda h: h['val_metrics']['loss'])
            cfg = meta['config']
            require(state['status'] == 'complete', 'Incomplete run')
            require([h['epoch'] for h in history] == list(range(1, len(history) + 1)), 'History gap')
            require(state['last_completed_epoch'] == len(history) == selected['epoch'] + 20, 'Stop mismatch')
            require(row['best_epoch'] == state['best_epoch'] == selected['epoch'], 'Best epoch mismatch')
            require(row['validation'] == selected['val_metrics'], 'Summary mismatch')
            require(state['best_val_loss'] == selected['val_metrics']['loss'], 'State CE mismatch')
            require(not row['pilot'] and row['seed'] == cfg['seed'], 'Identity mismatch')
            require(cfg['model'] == row['model'] == 'resnet50' and cfg['loss'] == row['loss'] == 'ce', 'Model mismatch')
            require(cfg['attention'] == row['attention'] == ('none' if row['variant'] == 0 else 'se'), 'Attention mismatch')
            require(all(h['train_images'] == 893 and h['val_metrics']['images'] == 112
                        and h['val_metrics']['threshold'] == .5 for h in history), 'Sample/threshold mismatch')
            records.append(dict(**row, completed_epochs=len(history), metadata=meta,
                                history=history, source_summary=str(source.relative_to(ROOT))))
    records.sort(key=lambda r: (r['seed'], r['variant']))
    require([(r['seed'], r['variant']) for r in records] == [(s, v) for s in (42, 43, 44) for v in (0, 3)], 'Expected six unique runs')
    common = ('imgsz', 'epochs', 'batch', 'workers', 'lr0', 'patience', 'optimizer', 'weight_decay', 'pretrained')
    for key in common:
        require(len({r['metadata']['config'][key] for r in records}) == 1, f'Config differs: {key}')
    for key in ('dataset', 'augmentation', 'preprocessing', 'source_hashes', 'python', 'torch'):
        require(len({json.dumps(r['metadata'][key], sort_keys=True) for r in records}) == 1, f'Metadata differs: {key}')
    expected = records[0]['metadata']['config']
    require(all(expected[k] == v for k, v in dict(imgsz=640, epochs=100, batch=4, workers=0,
                                                lr0=.001, patience=20).items()), 'Unexpected protocol')
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    dest = ROOT / 'reports/ablation/records' / ('repeats_' + now.strftime('%Y%m%d_%H%M%S_%f'))
    dest.mkdir(parents=True, exist_ok=False)
    for row in records:
        run = ROOT / row['run']
        row['artifacts'] = {}
        for name in ('run_config.json', 'history.json', 'continue_state.json', 'training_config.yaml', 'best.pt', 'last.pt'):
            path = run / name
            row['artifacts'][name] = dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=digest(path))
            if path.suffix != '.pt':
                target = dest / 'evidence' / f"seed{row['seed']}_{row['attention']}" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                require(digest(target) == digest(path), 'Copy verification failed')
    for source in sources:
        shutil.copy2(source, dest / 'evidence' / f'{source.parent.name}_summary.json')
    source_checks = {}
    for name, recorded in records[0]['metadata']['source_hashes'].items():
        path = CODE / 'src/oral_fluorescence_detection' / name
        source_checks[name] = dict(recorded=recorded, current=digest(path), matches=recorded == digest(path))
        shutil.copy2(path, dest / 'evidence' / name)
    shutil.copy2(Path(__file__), dest / 'evidence' / Path(__file__).name)
    metrics = ('loss', 'auroc', 'balanced_accuracy', 'sensitivity', 'specificity', 'accuracy')
    stats = {a: {k: dict(mean=statistics.mean(r['validation'][k] for r in records if r['attention'] == a),
                        sample_sd=statistics.stdev(r['validation'][k] for r in records if r['attention'] == a))
                 for k in metrics} for a in ('none', 'se')}
    pairs = []
    for seed in (42, 43, 44):
        b, s = [r for r in records if r['seed'] == seed]
        pairs.append(dict(seed=seed, **{k: s['validation'][k] - b['validation'][k] for k in metrics}))
    payload = dict(recorded_at=now.isoformat(), status='all_four_repeats_complete', results=records,
                   descriptive_statistics=stats, se_minus_baseline_by_seed=pairs, source_checks=source_checks,
                   primary_metric='minimum whole-validation unweighted unsmoothed CE',
                   uncertainty='Sample SD across 3 training seeds; not a clinical or population confidence interval.',
                   test_performance_computed=False, weight_storage='Original paths; hashes recorded, weights not copied.')
    (dest / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, metric, label in zip(axes, metrics[:3], ('Validation CE (lower is better)', 'Validation AUROC', 'Balanced accuracy')):
        for seed, color in zip((42, 43, 44), ('#0072B2', '#D55E00', '#009E73')):
            values = [r['validation'][metric] for r in records if r['seed'] == seed]
            ax.plot((0, 1), values, marker='o', color=color, label=f'Seed {seed}')
        ax.set_xticks((0, 1), ('ResNet50', 'ResNet50 + SE'))
        ax.set_title(label)
        ax.grid(axis='y', alpha=.2)
        if metric != 'loss': ax.set_ylim(0, 1)
        else: ax.set_ylim(0, .6)
    axes[0].legend()
    fig.suptitle('Same validation set, 112 images | CE-selected checkpoints | 3 training seeds')
    fig.savefig(dest / 'paired_comparison.png', dpi=180)
    fig.savefig(dest / 'paired_comparison.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for ax, row in zip(axes.flat, sorted(records, key=lambda r: (r['variant'], r['seed']))):
        h = row['history']
        ax.plot([v['epoch'] for v in h], [v['train_loss'] for v in h], label='Train CE', color='#0072B2')
        ax.plot([v['epoch'] for v in h], [v['val_metrics']['loss'] for v in h], label='Validation CE', color='#D55E00')
        ax.axvline(row['best_epoch'], color='#555555', linestyle='--', alpha=.6)
        ax.set_title(f"{'Baseline' if row['attention']=='none' else 'SE'} | seed {row['seed']} | best epoch {row['best_epoch']}")
        ax.set_xlabel('Epoch'); ax.set_ylabel('CE (panel-specific scale)'); ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Full recorded training histories; dashed line = minimum validation CE')
    fig.savefig(dest / 'training_histories.png', dpi=180)
    plt.close(fig)
    labels = dict(loss='验证CE', auroc='AUROC', balanced_accuracy='平衡准确率', sensitivity='敏感度', specificity='特异度', accuracy='准确率')
    lines = ['# ResNet50与SE三随机种子复验报告', '', f'记录时间：{now.isoformat()}', '',
             '## 结论', '',
             '四次新增复验均正常完成；结合已完成seed42，形成基线与SE各3次训练。三个种子中，SE的验证CE均低于基线，AUROC和平衡准确率均高于基线。支持将SE作为后续优化的优先候选，但尚不能证明独立数据上的泛化或临床有效性。', '',
             '## 实验设置与来源', '',
             '冻结v7整图二分类数据：训练893张（阴性候选332、阳性候选561），验证112张（43/69），测试112张。来源组计数为训练235、验证16、测试17；来源组未核验为独立患者。只使用train/val，本报告没有重新计算测试性能。',
             '采用ResNet50预训练模型，基线与增加SE比较；640像素全视野等比缩放/补边、缓存v1、几何在线增强、CE、AdamW、lr0.001、weight_decay0.0001、batch4、workers0、FP32、最多100轮、patience20。每组从预训练初始化重新训练；seed43基线→SE，seed44 SE→基线。',
             '按整体验证集无权重无平滑CE最低轮选择检查点；所有指标来自该轮，阳性类为1，阈值0.5。未改用AUROC最高轮。六组数据、缓存、增强、核心源码及记录的Python/PyTorch版本一致，具体指纹在results.json。',
             'seed42来自20261029_141726正式汇总（基线实际复用20261029_114509）；seed43来自20261030_174612，seed44来自20261030_193451。', '',
             '## 每次实验结果', '',
             '| 种子 | 方案 | 完成/最佳轮 | 验证CE | AUROC | 平衡准确率 | 敏感度 | 特异度 | 准确率 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['seed']} | {'基线' if r['attention']=='none' else 'SE'} | {r['completed_epochs']}/{r['best_epoch']} | " + ' | '.join(f"{r['validation'][k]:.4f}" for k in metrics) + ' |')
    lines += ['', '## 三种子描述性汇总', '', '以下为均值±样本标准差（n=3，ddof=1），标准差描述训练随机性，不是患者层面的置信区间。', '', '| 指标 | 基线 | SE | SE减基线（均值） |', '|---|---:|---:|---:|']
    for k in metrics:
        b, s = stats['none'][k], stats['se'][k]
        lines.append(f"| {labels[k]} | {b['mean']:.4f} ± {b['sample_sd']:.4f} | {s['mean']:.4f} ± {s['sample_sd']:.4f} | {s['mean']-b['mean']:+.4f} |")
    lines += ['', '![各种子配对比较](paired_comparison.png)', '', '| 种子 | ΔCE | ΔAUROC | Δ平衡准确率 |', '|---|---:|---:|---:|']
    for p in pairs:
        lines.append(f"| {p['seed']} | {p['loss']:+.4f} | {p['auroc']:+.4f} | {p['balanced_accuracy']:+.4f} |")
    lines += ['', 'Δ为SE减基线。CE差为负表示改善；比率指标差乘100为百分点。seed42提升最大，43/44幅度较小，不能只引用seed42的最佳值代表稳定性能。', '',
             '## 训练过程与混淆矩阵', '', '六次训练均在最佳轮后20轮无改善而结束。图中保留完整记录，虚线为选择轮；各面板纵轴独立，不能直接比较视觉斜率。', '',
             '![完整训练曲线](training_histories.png)', '',
             '训练诊断：seed43基线第26轮验证CE升至9.7464，seed44基线第32轮升至2.2515，均为实际历史值，图中未裁去。SE曲线后期也存在训练损失继续下降、验证损失波动/回升的现象。因此“所选检查点指标改善”不等于“全程训练已稳定”；应优先排查预测置信度、学习率及小批量归一化状态。仅凭曲线不能确定具体原因，后续诊断使用训练/验证数据。', '',
             '| 种子 | 方案 | TN | FP | FN | TP |', '|---|---|---:|---:|---:|---:|']
    for r in records:
        cm = r['validation']['confusion_matrix']
        lines.append(f"| {r['seed']} | {r['attention']} | {cm[0][0]} | {cm[0][1]} | {cm[1][0]} | {cm[1][1]} |")
    lines += ['', '## 解释边界与下一步', '',
             'SE是从此前八组探索中筛选出的候选，三个种子仍共用同一112张验证集。重复训练不会增加独立图像或患者数，也不能消除验证集反复选模造成的乐观偏差。未做显著性检验、患者/来源组bootstrap置信区间、分部位/模态推理或校准分析；本报告只陈述已记录结果。',
             '标签为正常/病变候选，未确认为临床诊断。已有测试集曾被历史评估，后续同集评估不能称全新独立盲测；最终研究确认需新的独立留出数据。',
             '建议下一阶段以SE为优先候选，在固定其他条件下比较0.0001、0.0003和0.001学习率，预先固定预算与选模规则。AMP吞吐评估单独进行，兼顾数值稳定性，不同时更改输入大小/损失/注意力。本次未启动任何新训练，也未冻结最终交付检查点。', '',
             '## 数据记录与复核范围', '',
             'results.json含六次完整指标/逐轮历史/配置/环境/数据与源码指纹、配对差值、样本标准差和12个best/last权重SHA-256；evidence含训练元数据副本与来源汇总。inventory.json覆盖本记录文件，并进行了回读校验。',
             '权重保留原实验目录，此处仅记录路径和哈希，不是含数据及权重的完整可迁移包。本次未反序列化权重或重新推理，哈希验证不等同模型加载验证。旧结果与原始数据保留。',
             '复现报告：在项目根目录运行 `.venv/Scripts/python.exe -B scripts/record_repeat_results.py`，会生成新的独立记录目录。']
    (dest / '复验报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    inventory = {p.relative_to(dest).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(dest.rglob('*')) if p.is_file()}
    (dest / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    require(all(digest(dest / p) == v['sha256'] for p, v in inventory.items()), 'Inventory verification failed')
    print(json.dumps(dict(report=str(dest), snapshot_files=len(inventory), stats=stats), ensure_ascii=True))


if __name__ == '__main__':
    main()
