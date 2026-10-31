"""Build a self-contained, aggregate-only PPT preparation handoff."""
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
package = ROOT / 'deliverables' / f'期中答辩_PPT同学交接包_{stamp}'
package.mkdir(parents=True, exist_ok=False)


def write(relative, content):
    target = package / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.strip() + '\n', encoding='utf-8-sig')


def copy(source, relative):
    target = package / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / source, target)


repeat = 'reports/ablation/records/repeats_20261030_214844_921038'
formal = 'reports/ablation/records/formal_20261030_161652_551258'
for name in ('paired_comparison.png', 'paired_comparison.svg', 'training_histories.png'):
    copy(f'{repeat}/{name}', f'03_图表素材/{name}')

copy(f'{formal}/结果报告.md', '04_证据报告/八组正式实验报告.md')
report = (ROOT / repeat / '复验报告.md').read_text(encoding='utf-8')
report = report.replace('](paired_comparison.png)', '](../03_图表素材/paired_comparison.png)')
report = report.replace('](training_histories.png)', '](../03_图表素材/training_histories.png)')
write('04_证据报告/三种子复验最终报告.md', report)

source_doc = ROOT / 'docs/期中答辩材料清单与模型构建思路_2026-10-31.md'
detail = source_doc.read_text(encoding='utf-8')
detail = detail.split('## 七、原始证据索引')[0]
detail += '''## 七、交接包证据索引

- [八组正式实验报告](../04_证据报告/八组正式实验报告.md)
- [三种子复验最终报告](../04_证据报告/三种子复验最终报告.md)
- [配对比较图](../03_图表素材/paired_comparison.png)
- [完整训练曲线](../03_图表素材/training_histories.png)
- [来源清单](../04_证据报告/来源清单.md)

原始代码、数据和权重保留项目内，本包供制作答辩材料使用。
'''
write('02_模型与数据说明/详细梳理.md', detail)

write('00_先读我.md', '''# 给PPT制作同学的交接说明

资料内容截至2026-10-31，实验结果截至2026-10-30。用途：制作本项目期中答辩PPT；这是资料包，尚不是完成的PPT。

## 建议阅读顺序

1. 打开[逐页内容与讲稿](01_逐页内容与讲稿.md)，可直接提取每页正文、讲解和版式建议。
2. 需要讲清网络时看[结构绘制说明](02_模型与数据说明/模型结构绘制说明.md)和[详细梳理](02_模型与数据说明/详细梳理.md)。
3. 从[图表使用说明](03_图表素材/图表使用说明.md)选择PNG或SVG；表格可从[可复制数据表](03_图表素材/可复制数据表.md)粘贴重排。
4. 数字有疑问，查04_证据报告中的两份报告。以最终三种子报告为主要结果依据。
5. 定稿前按[待补信息与检查清单](05_待补信息与定稿检查.md)核对。

## 整套PPT的中心句

已建立口腔白光/自体荧光整图候选二分类流程，完成八组筛选和基线、SE三种子复验；SE为后续优化的优先候选，独立泛化仍待验证。

## 最容易写错的五点

- 0.9138±0.0292是三个种子的验证AUROC，不是测试准确率。
- 来源组不是已核验独立患者；标签是正常/病变候选，不是临床确诊。
- 当前是一张图进入一个分类网络，没有白光与荧光配对双流融合。
- SE只接在ResNet50的layer4输出后，不是每个残差块里都有SE。
- Grad-CAM、最终模型冻结和推理交付均不能写成已完成。

## 文件与使用范围

本包包含汇总统计、训练曲线、文字说明，不包含患者原图、逐图预测、数据集、权重或运行环境。可以离开原项目目录阅读。图中的数值和坐标保留原意，重绘时勿删失败组或截掉损失异常。

学校模板、答辩时长、正式项目名称、成员与导师、经费情况需要项目负责人补充。背景页的临床数字和文献尚未在本次交接中核验，不要自行填入无来源数字。
''')

slides = [
('项目名称与阶段目标', '阶段目标：建立口腔整图候选分类流程，并筛选可继续优化的模型。\n正式项目名、学校、成员、导师按申报书补充。', '本次汇报重点是模型构建、实验设计和已取得的阶段证据，当前候选模型仍处于研究开发阶段。', '标题+一句目标，不写最终诊断系统已完成。', '项目负责人补正式信息'),
('研究问题与技术切入点', '输入：口腔白光或自体荧光完整图像。\n问题：能否通过迁移学习建立可复现的候选二分类流程？\n重点：保留视野、受控比较、复验与评价。', '我们把目标落实为图像级候选分类，先把数据和评价规则固定，再比较模型。背景临床意义需使用已核验文献补充。', '左侧应用问题，右侧三个技术重点；原图素材需负责人另行审核提供。', '详细梳理第1、3节'),
('阶段进度与技术路线', '已完成：数据接入、基线、八组筛选、四次新增复验。\n当前：ResNet50+SE+CE为优先候选。\n待完成：异常诊断、学习率对照、冻结评价与推理交付。', '正式缓存系列共12次训练，包括复用的seed42基线。八组筛选不是完整架构、注意力和损失的全部组合。', '三段进度条：完成→当前候选→后续工作；勿填虚构进度百分比。', '八组报告、复验报告'),
('数据与任务定义', '1117张：训练893、验证112、测试112。\n来源组235/16/17，独立患者关系待核验。\n类别：normal_candidate / lesion_candidate。\nXML框不参与训练，不裁ROI。', '同一来源组不跨集合，但这不等于已经完成患者级独立性证明。候选标签也不能直接替代病理或临床诊断。', '使用可复制数据表；底部保留标签和来源组解释。', '复验报告：实验设置与来源'),
('整图输入与训练流程', '缓存v1 → 几何在线增强 → 等比缩放补边 → 640×640 → ImageNet归一化。\n随机翻转与±10°旋转，旋转扩展画布。\n验证输入采用确定性处理。', '保留整幅视野，同时缩小分辨率以控制成本。缓存改变了训练变换顺序，因此作为独立版本重新训练；并非与旧流程完全等价。', '横向流程图；不画病灶裁剪、色彩校正或分割。', '详细梳理：输入处理；源码核对记录见来源清单'),
('ResNet50基线与SE候选', 'ImageNet预训练ResNet50，分类头改为2类。\nlayer4输出 → SE通道注意力 → 全局平均池化 → Linear(2048,2)。\n输出病变候选概率，阈值0.5。', 'SE对特征通道平均池化，经2048到128再到2048的映射生成通道权重，再乘回原特征。这里只增加一个模块，主干也参与训练。', '按照模型结构绘制说明重画为PPT原生形状；注意力分支使用另一颜色。', '模型结构绘制说明、ablation.py'),
('为什么设计这八组实验', '架构：ResNet50 / EfficientNet-B3 / DenseNet121。\n注意力：ResNet50的none / SE / CBAM。\n损失：无额外注意力ResNet50的CE / weighted CE / smoothing / Focal。\n固定其他条件，按最低验证CE选轮。', '每一类比较对应一个问题。并未验证SE和Focal组合，也没有分别为每个架构完成最优超参数搜索。', '三个并列问题框+统一设置脚注；完整设置放备份页。', '八组正式实验报告'),
('八组筛选：SE进入复验', 'SE+CE：验证CE 0.3362，AUROC 0.9447。\n按预定最低CE规则优先选择SE。\nweighted CE失败结果保留。', '这些都是seed42在验证集上的结果。weighted CE选中轮全部预测为阳性候选，不代表所有轮都如此；单种子赢家仍需要复验。', '展示八组表，突出SE行；表头必须写验证集、seed42。', '八组正式实验报告'),
('三种子复验：改善方向一致', '种子42/43/44：每个方案各3次训练。\n基线验证AUROC：0.8500±0.0434。\nSE验证AUROC：0.9138±0.0292。\nSE验证平衡准确率：0.8316±0.0343。', '三个种子中SE均改善，但seed42提升更大。误差为种子间样本标准差，同一112张验证集没有增加独立样本；这不是临床置信区间。', '优先放paired_comparison.svg，旁边放三行汇总；保留完整坐标和图例。', '三种子复验最终报告'),
('阶段问题与证据边界', '验证集较小且反复用于筛选。\n患者关系和临床标签待核验。\n损失尖峰与后期波动仍需诊断。\n当前SE尚无本轮测试性能报告。', '历史旧ResNet50测试AUROC为0.6943，属于旧设置，不能套到当前SE上。当前优先候选的验证改善不等于已经解决泛化。', '用问题—下一步对应表；完整六面板曲线放备份页，勿压缩成看不清的小图。', '最终复验报告；历史结果见详细梳理'),
('下一阶段：从候选到交付', '诊断异常 → 独立提速短测 → SE学习率对照 → 公平基线 → 冻结模型 → 分层评价与解释 → 推理交付。\n优先新增6次学习率训练，其余按条件扩展。', '先解决异常和公平性，再固定最终权重与评价方案。MIL和模态融合属于条件拓展，当前未实现。', '按依赖画流程，不把建议排期写成已承诺完成日期。', '详细梳理第五节'),
('阶段成果与团队工作', '形成可复现的整图分类流程。\n完成八组受控筛选和三种子复验。\n实现版本化缓存、恢复记录与训练看板。\n成员分工与论文/软件状态由负责人据实补充。', '当前成果主要是工程流程与探索性实验。SE是已有方法，不能单独当作原创算法；后续需要进一步验证和明确研究贡献。', '代码/报告/实验/后续交付四块；不展示患者姓名或私人路径截图。', '来源清单；负责人补充贡献'),
]
body = '# PPT逐页内容与讲稿\n\n默认12页主体，时长未知，先按内容制作，再按学校要求压缩。每页正文可直接重排；讲稿放备注区。\n\n'
for index, (title, bullets, notes, layout, evidence) in enumerate(slides, 1):
    body += f'## 第{index}页｜{title}\n\n### 页面文字\n\n'
    body += '\n'.join('- ' + line for line in bullets.splitlines())
    body += f'\n\n### 讲解备注\n\n{notes}\n\n### 排版与依据\n\n{layout}\n\n依据：{evidence}。\n\n'
body += '''## 如学校只允许8页

合并2/3页、5/6页、7/8页、11/12页；保留第4页数据、第9页复验、第10页局限性。不要通过删掉结果口径和失败记录压缩时间。

## 备份页建议

完整训练设置、六次混淆矩阵、完整训练曲线、SE公式、历史测试与当前验证的区别、候选标签说明。答辩问答见详细梳理第六节。
'''
write('01_逐页内容与讲稿.md', body)

write('02_模型与数据说明/模型结构绘制说明.md', '''# 模型结构绘制说明（可用PPT原生形状重画）

## 主图：从左到右

单张完整RGB图 → 输入处理 → 3×640×640 → ResNet50 stem → layer1 → layer2 → layer3 → layer4 → SE → 全局平均池化 → 2048维向量 → 两类线性层 → Softmax → 候选概率。

基线图删除SE，其余结构保持。ResNet50采用ImageNet预训练初始化，分类头替换为两类，主干共同训练。不要画成检测框输出。

## SE局部放大图

输入特征F从左侧分两支：

- 上支：全局平均池化 → 1×1卷积2048→128 → ReLU → 1×1卷积128→2048 → Sigmoid → 通道权重s。
- 下支：原特征F直接保留，和s逐通道相乘，输出F'。

SE只接在layer4整体输出后、主干原有全局平均池化之前。不要给所有残差块加SE，不要把权重画成病灶分割掩膜。

公式：z_c=mean(F_c)，s=sigmoid(W2·ReLU(W1·z))，F'_c=s_c·F_c。

640输入下末端特征约为2048×20×20，这是标准网络尺寸推导，不是本次新增实测记录；PPT可以仅标通道数，减少不必要细节。

## 模态关系图

白光单图 ─┐
          ├→ 同一个单图分类器 → 两类候选概率
荧光单图 ─┘

两类图分别独立进入网络，不同时配对输入。当前不包含双流融合、模态嵌入、MIL或临床确诊模块。
''')

formal_text = (ROOT / formal / '结果报告.md').read_text(encoding='utf-8')
formal_table = formal_text[formal_text.index('| 编号'):formal_text.index('\n\n最低CE')]
repeat_table_start = report.index('| 指标 | 基线')
repeat_table = report[repeat_table_start:report.index('\n\n![', repeat_table_start)]
per_run_start = report.index('| 种子 | 方案 | 完成')
per_run = report[per_run_start:report.index('\n\n##', per_run_start)]
write('03_图表素材/可复制数据表.md', f'''# 可复制数据表

## 数据计数

| 集合 | 图像 | 正常候选 | 病变候选 | 来源组 |
|---|---:|---:|---:|---:|
| 训练 | 893 | 332 | 561 | 235 |
| 验证 | 112 | 43 | 69 | 16 |
| 测试 | 112 | 43 | 69 | 17 |

来源组尚未核验为独立患者；XML不参与训练。

## 八组筛选（seed42，验证集112张）

{formal_table}

## 三种子汇总（同一112张验证集）

{repeat_table}

均值±样本标准差，n=3，ddof=1；不是患者置信区间，不是测试性能，也不是集成性能。

## 六次训练（验证集）

{per_run}

共同设置：640、batch4、workers0、AdamW、lr0.001、weight_decay0.0001、FP32、最多100轮、patience20；按最低无权重无平滑验证CE选轮，阈值0.5。
''')

write('03_图表素材/图表使用说明.md', '''# 图表使用说明

| 文件 | 放在哪页 | 推荐图注 |
|---|---|---|
| paired_comparison.svg / .png | 第9页主图，SVG适合放大 | 基线与SE在三个训练随机种子上的配对比较；均为同一112张验证集结果。 |
| training_histories.png | 备份页；需要讲波动时单独整页 | 六次训练的完整历史，虚线为所选检查点；各面板纵轴独立。 |
| 可复制数据表.md | 第4、7、8、9页 | 具体数据口径见各表标题。 |

优先使用SVG，兼容性不足时用PNG。图中文字过小时重新布局页面，不要截断图例、坐标、异常峰值或只留最有利的一个种子。三种子汇总必须保留“验证”“均值±标准差”。

本包不含患者图像、Grad-CAM、推理软件截图。若需要这些素材，请由项目负责人确认实际完成状态、授权与脱敏后提供。不得用示意图伪装成实验结果。
''')

write('05_待补信息与定稿检查.md', '''# 待补信息与定稿检查

## 项目负责人补充

- 学校正式中期通知、PPT模板、答辩时长、截止时间。
- 原申报书中的正式名称、目标、计划节点及与当前路线的对应关系。
- 成员、导师、学院等正式信息，真实分工和贡献。
- 经费预算/执行状态、真实成果证明、导师指导记录。
- 背景页经核验的文献和临床事实；本次未新增文献检索。
- 如需项目指导书背景，可在原项目根目录找到DOCX，由负责人核对后提取；本包未直接打包，避免混入未复核口径或个人信息。

## 逐项定稿检查

- [ ] AUROC、准确率、平衡准确率没有混用；验证结果未写成测试结果。
- [ ] 主结论以三种子复验为依据；0.9447注明seed42单次验证值。
- [ ] ±表示训练种子间样本标准差，没有写成95%置信区间。
- [ ] 未将来源组数量写成患者数量，未将候选类别写成临床确诊。
- [ ] SE接入位置正确，没有画成全残差块改造或模态双流融合。
- [ ] 当前分类指标没有与旧YOLO的mAP混用。
- [ ] 失败组和训练波动有解释，未写“全程训练稳定”。
- [ ] 计划中的调参、Grad-CAM、冻结与软件没有写成已完成。
- [ ] 文献、论文、专利、软著、经费和会议记录没有补造。
- [ ] 旧测试AUROC 0.6943单列为历史旧基线，未配到当前SE下。
- [ ] 没有带患者姓名、原图、个人文件路径的截图或逐图预测。
- [ ] 项目负责人及导师复核科学结论，PPT制作同学确认排版和字体。
''')

sources = {
    '详细梳理': 'docs/期中答辩材料清单与模型构建思路_2026-10-31.md',
    '八组报告': f'{formal}/结果报告.md',
    '复验报告': f'{repeat}/复验报告.md',
    '总体规划': 'docs/模型训练与优化总体规划_2026-10-30.md',
    '模块实现': 'experiments/oral_v7_resnet18_baseline_20261026/code/src/oral_fluorescence_detection/ablation.py',
    '训练实现': 'experiments/oral_v7_resnet18_baseline_20261026/code/src/oral_fluorescence_detection/image_classification.py',
    '增强实现': 'experiments/oral_v7_resnet18_baseline_20261026/code/src/oral_fluorescence_detection/online_augmentation.py',
}
source_text = '# 来源清单\n\n原项目内的相对路径及SHA-256，供负责人追溯；不需要这些原始路径也能阅读本包。未将逐图数据、运行环境或原始JSON整体带入本包。\n\n'
for label, relative in sources.items():
    digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    source_text += f'- {label}：`{relative}`\n  SHA-256：`{digest}`\n'
write('04_证据报告/来源清单.md', source_text)

for markdown in package.rglob('*.md'):
    for match in re.finditer(r'\]\(([^)]+)\)', markdown.read_text(encoding='utf-8-sig')):
        if not (markdown.parent / match.group(1)).is_file():
            raise ValueError(f'Broken package link: {markdown.name}: {match.group(1)}')

inventory = {}
for item in sorted(package.rglob('*')):
    if item.is_file():
        inventory[item.relative_to(package).as_posix()] = {
            'bytes': item.stat().st_size,
            'sha256': hashlib.sha256(item.read_bytes()).hexdigest(),
        }
write('文件清单与校验.json', json.dumps(inventory, ensure_ascii=False, indent=2))
archive = package.with_suffix('.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    for item in sorted(package.rglob('*')):
        if item.is_file():
            bundle.write(item, item.relative_to(package.parent))
with zipfile.ZipFile(archive) as bundle:
    if bundle.testzip() is not None:
        raise ValueError('Archive integrity verification failed')
    for relative, expected in inventory.items():
        payload = bundle.read(f'{package.name}/{relative}')
        if hashlib.sha256(payload).hexdigest() != expected['sha256']:
            raise ValueError(f'Archive checksum mismatch: {relative}')
print(json.dumps({'folder': str(package), 'zip': str(archive),
                  'files': len(inventory) + 1, 'bytes': archive.stat().st_size,
                  'links': 'all local links verified', 'zip_check': 'passed'}, ensure_ascii=False))
