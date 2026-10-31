"""One-shot local final report after the specified architecture queue terminates."""
import argparse
import json
import time
from pathlib import Path
import psutil
from run_v8_architecture_queue import verified_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    report=Path(args.report).resolve()
    while True:
        queue=json.loads((report/'queue.json').read_text(encoding='utf-8'))
        status=queue['status']
        if status!='running' or not args.watch:
            break
        try:
            proc=psutil.Process(queue['pid'])
            alive=proc.is_running() and abs(proc.create_time()-queue['process_created'])<.1
        except psutil.Error:
            alive=False
        if not alive:
            status='interrupted'
            break
        time.sleep(15)
    rows=[]
    for row in queue['completed']:
        run=Path(row['run'])
        meta=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
        checked=verified_result(run,row['model'],Path(meta['data']).resolve(),Path(meta['cache']).resolve(),row['seed'],row['reused'])
        if checked!=row:
            raise ValueError('Saved queue result differs from verified training evidence')
        rows.append(checked)
    lines=['# v8三架构同条件验证报告','',
        f"队列状态：{status}；已完成{len(rows)}/3组。仅seed42验证集结果，无测试性能评价。",'',
        '固定条件：train782、val223，640像素、batch4、FP32；20轮分类头+40轮末端两阶段+全网络微调至早停。学习率依次1e-3/1e-4/1e-5，Adam L2=1e-4，Dropout0.5，训练类别加权CE+标签平滑0.1。', '',
        '全阶段最低验证普通未加权CE选择检查点；阈值0.5。训练损失与验证CE定义不同。','',
        '| 架构 | 训练轮数 | 最佳轮 | 验证CE | AUROC | 平衡准确率 | 灵敏度 | 特异度 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['validation']
        values=[m[k] for k in ('loss','auroc','balanced_accuracy','sensitivity','specificity')]
        fmt=['NA' if v is None else f'{v:.4f}' for v in values]
        lines.append('| '+ ' | '.join([row['model'],str(row['epochs']),str(row['best_epoch']),*fmt])+' |')
    if status=='complete' and len(rows)==3:
        candidate=min(rows,key=lambda r:r['validation']['loss'])
        lines+=['',f"按预定最低验证CE规则，本轮候选架构为{candidate['model']}。这是同一验证集、单seed的探索性选择，尚不能宣称稳定优于其他架构。"]
    else:
        lines+=['','队列未全部完成，不给出完整架构排名或最终候选结论。',str(queue.get('error',''))]
    lines+=['', 'ResNet-50复用已核验兼容的v8完整记录；其余模型从各自通用ImageNet权重重新初始化。旧v7结果仅作为方法来源，不计入本表。', '',
        '候选标签不是临床确诊，来源组不是确认独立患者。重新划分不能消除历史测试暴露，也不构成全新独立盲测；多seed复验仅可评估训练波动。', '',
        '证据：queue.json、validation_summary.json、evidence源码和元数据快照；best/last权重哈希保存在队列记录，权重保留各run目录。']
    target=report/'架构验证报告.md'
    if target.exists():
        raise FileExistsError('Preserve existing report; choose a separate report version')
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Report written: {target}',flush=True)


if __name__=='__main__':
    main()
