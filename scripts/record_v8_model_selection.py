"""Verify completed v8 evidence and archive a validation-only model decision."""
import csv
import json
from datetime import datetime
from pathlib import Path

from run_v8_architecture_queue import ROOT, read, verified_result


def main():
    report = Path(read(ROOT / 'reports/_qa/v8_architecture_queue.json')['report'])
    queue = read(report / 'queue.json')
    assert queue['status'] == 'complete' and len(queue['completed']) == 3
    rows = []
    for row in queue['completed']:
        run = Path(row['run'])
        meta = read(run / 'run_config.json')
        checked = verified_result(run, row['model'], Path(meta['data']).resolve(),
                                  Path(meta['cache']).resolve(), row['seed'], row['reused'])
        if checked != row:
            raise ValueError('Queue evidence differs from actual run')
        rows.append(checked)
    if read(report / 'validation_summary.json') != rows:
        raise ValueError('Summary differs from verified evidence')
    chosen = min(rows, key=lambda r: r['validation']['loss'])
    target = report / ('selection_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    target.mkdir(exist_ok=False)
    decision = dict(created=datetime.now().astimezone().isoformat(),
                    status='selected_for_next_optimization',
                    rule='minimum validation ordinary unweighted unsmoothed CE across best checkpoints',
                    selected_model=chosen['model'], checkpoint=str(Path(chosen['run']) / 'best.pt'),
                    checkpoint_sha256=chosen['weight_sha256']['best.pt'],
                    best_epoch=chosen['best_epoch'], selection_split='validation',
                    test_evaluated=False, dataset_sha256=queue['dataset_sha256'],
                    cache_manifest_sha256=queue['cache_manifest_sha256'],
                    source_report=str(report), results=rows,
                    next_stage='stability replication before attention/loss ablations; no new run launched')
    (target / 'selection.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    keys = ['model', 'epochs', 'best_epoch', 'loss', 'auroc', 'accuracy',
            'balanced_accuracy', 'sensitivity', 'specificity']
    with (target / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] if k in row else row['validation'][k] for k in keys})
    lines = ['# v8架构选择与后续阶段交接记录', '',
             f"记录时间：{decision['created']}。三组训练、阶段历史、早停、数据/缓存指纹及best/last权重哈希均重新核验通过。", '',
             '**选择DenseNet-121作为后续优化基础架构；使用第39轮best.pt，不使用第80轮last.pt。**', '',
             '选择规则沿用实验前约定：全阶段最低验证普通、未加权、无标签平滑CE。下表为同一223张验证图像、seed42、阈值0.5的各自最佳检查点结果。', '',
             '| 模型 | 总轮数 / 最佳轮 | CE | AUROC | 准确率 | 平衡准确率 | 灵敏度 | 特异度 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        v = row['validation']
        lines.append(f"| {row['model']} | {row['epochs']} / {row['best_epoch']} | {v['loss']:.4f} | {v['auroc']:.4f} | {v['accuracy']:.2%} | {v['balanced_accuracy']:.2%} | {v['sensitivity']:.2%} | {v['specificity']:.2%} |")
    lines += ['',
        'DenseNet验证CE最低，AUROC和平衡准确率也最高。相较B3，CE低0.03798、平衡准确率高0.83个百分点，AUROC仅高0.000514；灵敏度低0.72个百分点、特异度高2.38个百分点。该差异未做显著性检验，不能宣称稳定优于B3。ResNet灵敏度最高，但特异度为75%，其余综合指标较低。当前没有实测推理延迟/峰值显存，选择不包含部署效率结论。', '',
        'DenseNet混淆矩阵（行是真实正常/病变候选，列为预测正常/病变候选）：[[75,9],[12,127]]。B3为[[73,11],[11,128]]，ResNet为[[63,21],[6,133]]。', '',
        '## 可复用权重与代码', '',
        f"- 最佳权重：`{decision['checkpoint']}`。",
        f"- SHA256：`{decision['checkpoint_sha256']}`。",
        '- 架构及阶段映射：scripts/v8_architecture_support.py；训练入口：scripts/train_v8_architecture.py。',
        '- DenseNet末端denseblock3/4、transition3及norm5参与第二阶段；分类头Dropout0.5后接二分类Linear。没有额外SE或白光/荧光配对双流。',
        '- DenseNet发生过进度写入错误，从完整第14轮状态恢复；第15轮训练损失重放一致，最终80轮完成。I/O适配器、失败快照及恢复轨迹保留在原报告目录。', '',
        '## 后续阶段', '',
        '1. 优先复验稳定性：建议DenseNet及最接近的B3使用相同新增seed（如43、44），保留本次42，共三seed；各自从约定ImageNet通用预训练初始化，保持v8划分、缓存及4.3协议。报告每seed和均值/标准差，再决定是否保留架构选择。当前尚未启动这些新增训练。',
        '2. 确认架构后开展注意力或损失函数消融，每次改变一个因素；本次DenseNet固定为可追溯对照。新增结构或损失实验需重新进行公平训练，不能加载本次best.pt后当作相同初始化比较。',
        '3. 当前best.pt可直接用于验证集错误分析、解释图及软件集成检查；如开展新任务微调，应另建实验并明确继承此权重。阈值选择仅使用验证集。', '',
        '## 记录边界', '',
        '固定条件：train782/val223，640像素、batch4、FP32、Adam、L2=1e-4、Dropout0.5、类别加权CE及标签平滑0.1；20轮头部lr1e-3、40轮末端阶段lr1e-4、全网络lr1e-5至本阶段验证CE连续10轮不改善。前两阶段固定轮数，最佳检查点允许来自任一阶段。', '',
        '本次仅选择下一阶段候选架构，未运行新的训练或测试集评价。单seed同验证集结果不能证明泛化或临床有效性；标签为候选标签，来源组不是确认独立患者。旧图像重新划分存在历史测试暴露，多seed不会增加独立样本。', '',
        '同目录selection.json保存完整指标、原始路径与权重哈希；comparison.csv可用于表格和PPT。原始队列及已生成报告保持不变。']
    (target / '模型选择与后续阶段.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log = ROOT / 'experiments/oral_v7_resnet18_baseline_20261026/code/docs/PROJECT_WORK_LOG.md'
    with log.open('a', encoding='utf-8') as handle:
        handle.write('\n## 2026-10-31 v8架构结果归档与候选选择\n\n'
                     '- 三架构均已完成并重新核验完整阶段、早停、数据/缓存及权重哈希；ResNet73轮最佳37，B3共78轮最佳55，DenseNet80轮最佳39。\n'
                     '- 按预定最低验证普通CE规则选DenseNet-121，CE0.2434586092、AUROC0.9610311751、平衡准确率0.9032631038、灵敏度0.9136690647、特异度0.8928571429；使用第39轮best.pt。单seed探索性选择，未宣称稳定显著优于B3。\n'
                     f'- 完整决策、指标CSV及可复用权重清单：{target.relative_to(ROOT).as_posix()}。建议下一阶段先对DenseNet/B3配对seed复验，再单因素消融；本次没有启动新增训练、测试集评价或Git提交。\n')
    print(json.dumps({'record': str(target), 'selected_model': chosen['model'],
                      'checkpoint': decision['checkpoint'], 'verified_runs': len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
