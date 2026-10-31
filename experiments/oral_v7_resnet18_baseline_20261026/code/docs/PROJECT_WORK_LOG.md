# Work log for the repository source snapshot

## 2026-10-28 GitHub source update

- Preserved the existing whole-image source/configuration/test files as a Git snapshot without moving or changing the training implementation.
- Included the pooled-training and AI handoff protocols; excluded raw/processed data, model weights, training outputs, caches and private evidence.
- Root binary pipeline tests: four passed using `.venv/Scripts/python.exe -m unittest discover -s tests -v`.
- Nested test discovery failed: pytest is absent from the current environment, and `scripts/build_whole_image_classification.py` is missing from this snapshot. No claim of full reproducibility or new model validation is made.
- Root README records the snapshot location, missing input package and test limitations. Dataset labels, splits and images were not changed.
- The earlier source project's work log is not present here; this entry records only this repository session and does not reconstruct its history.
- Remote main contains two additional documentation commits; preserve them when integrating the local update before pushing.

## 2026-10-28 Architecture, attention and loss ablations

- Added torchvision EfficientNet B0/B3 and DenseNet121 alongside existing ResNets; optional terminal feature-map SE/CBAM; CE, training-count weighted CE, label-smoothed CE and focal loss. Validation model selection remains ordinary overall cross-entropy.
- Saved attention/loss metadata in checkpoints and rebuilt the selected architecture during explicit evaluation. New continuation scripts use the saved training YAML.
- Added eight controlled variants, pilot mode (320, one epoch), formal mode (640, up to 100 epochs), PyCharm .run configurations and a read-only live result monitor. No frozen data or split changes.
- Root unittest suite: 7 tests passed, including 12 architecture/attention combinations with gradients/reload and loss checks. Synthetic epoch-boundary resume check: histories, model and optimizer identical. No clinical interpretation of synthetic tests.
- Initial pilot was stopped before completing its first epoch due to slow CPU preparation; restarted with four torch CPU threads. Partial output remains preserved as an incomplete run. Real-data pilot results pending in reports/ablation; test performance is not calculated.
- User requested PyCharm presentation. Desktop controller denied access to PyCharm; configuration files and instructions were prepared without claiming UI interaction succeeded.

- User chose to start experiments manually in PyCharm. Stopped the restarted pilot before any completed-epoch result; preserved partial run directories. No completed real-data ablation results are claimed.

## 2026-10-28 Correction: ResNet50 ablation baseline

- User confirmed ResNet50 as the formal ablation baseline. Changed variants 0/3/4/5/6/7 from ResNet18 to ResNet50 in the runner, PyCharm configuration names and usage guide; EfficientNet-B3 and DenseNet121 remain architecture controls.
- All new variants retain batch 4 and identical training settings. Historical ResNet50 used batch 8; variant 0 must be rerun for a matched comparison. No historical results or incomplete pilot artifacts were changed.
- Added ResNet50 to forward/backward/reload tests, including none/SE/CBAM. Seven unittest tests passed, covering 15 architecture/attention combinations. Verified all eight experiment-to-PyCharm mappings and parsed all ten XML configurations.
- No training was started; user will launch in PyCharm.

## 2026-10-28 Restart interrupted pilot at variants 5-7

- User requested restart preserving progress. Kept five completed results (variants 0-4) in reports/ablation/20261028_134000; variant 5 had no best.pt/last.pt, so its unfinished first epoch cannot be resumed.
- Stopped the two verified old pilot process IDs 37480/39000. Started a hidden local process for --pilot --variants 5 6 7, preserving seed, data, batch, resolution and epoch budget. Original partial outputs remain intact.
- Added train progress logging at batch 1, each 25 batches and the last batch; no optimization or data changes. Restart metadata and console logs: reports/ablation/restart_20261028_5_to_7/. This restart uses a local process, not the inaccessible PyCharm UI.


## 2026-10-28 Live training visualization

- Added a read-only Python standard-library dashboard server and responsive local page. Binds only to 127.0.0.1; serves dashboard and aggregated status only, without exposing arbitrary workspace files or data packages.
- Displays restarted pilot batch progress, cumulative within-epoch training loss, completed validation metrics, and source logs. Reads existing summaries to preserve variants 0-4 and accepts only matching ResNet50-based pilot settings (seed 42, size 320).
- Verified HTTP endpoints, JSON aggregation and 404 for unrelated paths. Opened the dashboard in the in-app browser and visually verified the chart, 5/8 completed groups and 200/224 training batches for variant 5. Page refreshes every three seconds; no changes to the running trainer.
- Dashboard URL is recorded in reports/ablation/restart_20261028_5_to_7/dashboard_url.json. Progress reflects logged batches and is not independent process-liveness proof.


## 2026-10-28 试跑数据正式记录

- 已核对8组全部完成：原运行保留编号0-4，重启运行保留编号5-7。每组1轮、320输入、batch 4、seed 42，893张训练、112张验证；没有计算测试集性能。
- 归档目录：reports/ablation/records/20261028_193507_083466/。记录报告.md提供8组指标表；results.json保留完整训练损失、验证指标、混淆矩阵、配置、数据指纹、训练源码指纹与权重SHA-256；evidence保存原配置、历史和进度日志副本。
- 校验：8组身份匹配、训练历史与验证汇总一致、数据SHA-256一致、全部16份权重文件存在且已计算哈希；归档inventory逐项复验通过。未复制原始数据或逐图预测。
- 记录边界：单轮结果不代表架构优劣；编号0/4/6在阈值0.5下全部预测阳性，阴性结果原样保留。前五组无逐批日志，不能补造曲线。

## 2026-10-28 正式训练启动准备（PyCharm）

- 用户要求正式训练仍从PyCharm启动。已确认没有遗留训练进程，新增 `.run/ablation_formal_all.run.xml`：全部8组，640、batch 4、最多100轮、patience 20，沿用冻结数据与验证CE选模，不计算测试性能。
- 验证XML可解析、参数无--pilot、无单组过滤且保留数据入口。更新操作说明；未启动任何后台正式训练。
- PyCharm桌面访问再次返回未批准，不能点击运行；用户需从PyCharm选择正式配置运行，或从原配置参数中删除--pilot。旧看板仍为试跑数据，不能声称已显示正式训练。

## 2026-10-28 正式训练可视化

- 用户已从PyCharm启动正式8组训练，当前批次reports/ablation/20261028_194751。只读检查确认ResNet50完成25轮；最低验证CE位于第22轮（0.469664），对应验证AUROC 0.904280；第25轮AUROC 0.819009。未计算测试性能。
- 看板新增--formal-report模式，按指定正式批次读取history.json和continue_state.json，显示每轮训练/验证损失、每轮AUROC、完成轮数和实验队列；保留旧试跑看板与结果。
- 正式看板地址保存在对应批次dashboard_url.json；HTTP和数据模式校验通过，浏览器截图确认双损失曲线已显示。每3秒读取，每轮结束更新；无法从已运行的PyCharm进程事后获得逐批持久日志，因此没有虚构轮内进度，也未重启或修改训练进程。


## 2026-10-28 正式训练GPU与速度检查

- 用户询问GPU使用与速度。训练进程69184在nvidia-smi中列为GPU进程，配置device=0；采样5次利用率2%-5%，GPU总显存约5755MiB（包含桌面及其他程序，不能视为训练独占），功耗约25W。进程在约10秒窗口消耗16.375 CPU秒。
- 已完成27轮、下一轮28；640输入、batch4、workers0，代码无AMP、无pin_memory与异步传输。源图在CPU解码/旋转后再缩放。
- 独立小样本只读预处理计时：24张640输出，7.692秒，约3.12张/秒；它不是完整profile，不能据此宣称精确瓶颈占比或保证提速倍数。系统可见内存15.46GiB，可用约1.97GiB。
- 未修改或重启正在运行的训练；建议后续先做workers=2的数据吞吐基准，再评估pin_memory和AMP，避免随意扩大batch或改变图像处理顺序。


## 2026-10-29 交付图片尺寸与训练预处理核查

- 只读检查当前完整交接包的先看这里、图片README、验收说明及全部1117张JPG文件头；没有进行全量像素完整性/哈希复验或测试性能评估。全部文件头可读取，实际宽高均与dataset.json逐条记录一致，640×640为0张，最长边不超过640也为0张。
- 常见尺寸为5184×3456共546张、1920×1080共339张、1080×1920共162张、3456×5184共44张、2976×1984共25张、4096×3072共1张。JPEG共2024.05MiB；按原尺寸三通道uint8估算，全体解码缓冲区32.88GiB，训练集26.97GiB，未含增强中间数组。
- 图片README明确原始图片字节保持不变、未裁剪/重新编码；manifest的image_size=640不能当作现存图片尺寸。当前Dataset每次读取JPG转RGB，先在原尺寸随机增强，再等比缩放补边和归一化；没有解码缓存。
- 澄清此前“每次都需要读取/解码”的表述：可通过缓存避免重复JPG解码；验证集可精确缓存确定性输入。训练若改为先缩小缓存再随机旋转，会改变插值与变换顺序，应作为有记录的优化版本验证，而非宣称等价。原尺寸训练集全缓存在当前16GB级主机不合适。未修改数据包、训练配置或正在运行的训练。


## 2026-10-29 已审核缓存方案实现与验证

- 用户批准先实现缓存并测速，保持现有正式训练。新增image_cache.py、scripts/prepare_image_cache.py，缓存目录data/cache/oral_binary_640_v1已加入Git忽略；不覆盖冻结包，不生成测试缓存，白光/荧光按模态分别存储。
- train893张缓存最长边640的uint8整图，后续在线随机增强再缩放补边归一化；val112张缓存原流程float32输入。缓存1,232,037,120字节（1.147GiB），清单记录源清单/源图片/负载/实现哈希、标签和划分；首次加载校验，失效时报错并要求新版本，不原地覆盖。缓存按需映射读取，不强制全量内存常驻。
- 训练入口新增可选cache_dir，默认原流程保持兼容。实验/检查点记录预处理版本和缓存清单哈希，拒绝跨预处理版本原地续训。ablation入口支持--cache-dir、--workers；新PyCharm配置独立ResNet50/none/CE、640、batch4、workers0，旧配置不变。新增只验证/测速配置与docs/PyCharm缓存训练使用说明.md。
- 全1005张缓存输出形状和有限数检查通过；112张验证输入与原流程bitwise一致；标签/模态/split记录一致，源train/val图片SHA-256核对通过。训练入口check_only通过，没有构建GPU模型或训练。缓存相关3项单元测试通过，涵盖验证完全一致/无需JPG、损坏及失效拒绝、在线增强随机性与种子复现；现有二分类流程4项测试通过。Python语法、两份PyCharm XML和git diff --check通过（只有既有CRLF提示）。
- 相同64张训练样本、batch4、torch CPU线程2、每组两次；原流程workers0约3.17-3.25张/秒、workers2约3.91-4.32；缓存workers0约30.23-36.82、workers2约10.59-11.63。资源采样进程树RSS峰值约958/2243/607/1613MiB（上述四组，RSS可能重复计共享页）。无资源采样首轮缓存workers0约43.80-45.53。短样本含Windows启动/退出，不能宣称完整训练提速倍数或证明整轮workers0一定更快；当前低内存下选workers0作为起点。
- 证据reports/cache/640_v1_verification.json、640_v1_resources.json、cache_preflight.json及缓存验证与测速结果.md。没有运行新流程GPU模型对照或评价模型效果，未计算测试性能。现有正式进程69184保持存活；只读状态检查已完成47轮。新增配置尚未启动，后续用户审核实测结果再切换独立实验。所有源码改动供后续运行使用，当前已加载训练进程继续原流程。


## 2026-10-29 用户授权切换缓存优化训练

- 用户要求立即使用优化形式开始训练，沿用PyCharm启动要求。使用computer-use技能读取并激活唯一PyCharm项目窗口；本次UI访问成功。通过PyCharm停止原训练（界面显示KeyboardInterrupt，属于主动停止），未删除旧实验。独立读取last.pt确认完整第58轮、58条历史且包含优化器状态，best.pt存在；第59轮尚未保存的批次不算已完成进度。
- 原训练进程退出后，从PyCharm的Alt+Shift+F10选择Ablation - Cache v1 ResNet50 (640, workers 0)运行。确认新launcher81148父进程35128为PyCharm，训练子进程83304，实际命令含--cache-dir、--workers 0、--variants 0。独立批次reports/ablation/20261029_114509，运行名ablation_20261029_114509_0_resnet50_none_ce_cache_v1；640、batch4、CE、AdamW、最多100轮及patience20，预处理版本写入run_config；从预训练权重新起，不混接旧第58轮。
- PyCharm界面确认实际训练批次输出，新流程完成训练与验证并保存best/last/history/continue_state；最新检查完成2轮。首轮从报告目录创建11:45:09至历史保存11:46:20约71秒（包括启动预检初始化，非纯每轮耗时）。GPU进程列表包含83304，两次利用率采样88%及74%、总显存约5154-5165MiB（非训练独占），不能宣称持续利用率或严格相对加速倍数。未运行测试推理或计算测试性能。
- 原看板进程经端口与命令核对后替换为新批次，同一13949端口；新增plan.json记录仅1组缓存实验，看板按计划显示实验数及缓存优化标识，避免误报8组待运行。浏览器刷新与截图确认新实验曲线、2/100轮（截屏时）和0/1实验数。旧训练与历史报告保持，切换证据reports/cache/switch_20261029，截图reports/cache/optimized_training_started.png；更新PyCharm缓存使用说明。

## 2026-10-29 训练完成状态核查

- 用户询问是否所有训练已结束。只读进程检查未发现run_architecture_ablation或train_image_classifier训练进程；当前缓存批次plan.json仅含variant0，validation_summary.json存在且continue_state标记complete。
- 缓存ResNet50/none/CE完成65轮；最低验证CE位于45轮（0.4903618852），对应AUROC0.8021570610。最佳轮后20轮未改善，按patience20提前停止；history最近保存时间12:46:19。看板返回1组缓存优化训练全部完成。
- 结论：本次单组正式训练已正常结束，不能表述为全部8组正式消融均完成。旧原流程ResNet50此前已在58轮主动停止；其余7组仅有既往单轮试跑，不是本次缓存正式实验。未启动后续训练或计算测试性能。


## 2026-10-29 从PyCharm启动八组缓存正式实验队列

- 用户明确要求开始八组正式实验并仍由PyCharm启动。确认没有遗留训练进程；复用已完成的缓存第0组ResNet50/none/CE（65轮、最佳45轮），其余七组依次新训练。没有把试跑或旧原图流程结果混入正式队列。
- runner新增--reuse-baseline，检查正式配置、seed、workers、data manifest哈希、缓存manifest哈希、online_augmentation源码哈希、complete状态与best.pt存在，并从实际历史重新生成验证汇总；新报告明确reused_from。新增plan.json用于显示完整8组队列。实际基线匹配预检成功，错误seed被拒绝；不构建模型的预检完成后才启动UI。首次补丁上下文不匹配未生效，已调整上下文完成修改。
- 新增.run/ablation_cache_formal_eight.run.xml及使用说明。从唯一PyCharm窗口通过Alt+Shift+F10选择Ablation - Cache Formal Eight (reuse completed baseline)并运行；确认启动进程86896的父进程为PyCharm35128，训练子进程94248，命令含cache-dir、workers0和reuse-baseline。不是终端后台启动训练。
- 新批次reports/ablation/20261029_141726，plan包含0..7，validation_summary已包含复用第0组。PyCharm实际输出Starting variant 1: efficientnet_b3/none/ce及第1轮第1/25批次，GPU单次采样99%及总显存7640MiB（包含其他程序，非训练独占）。训练正在运行，未声称后续6组已执行或8组已完成。
- 所有新组使用相同640、batch4、workers0、seed42、AdamW及缓存预处理，最多100轮、patience20；只用冻结train/val，不计算测试性能。依次执行EfficientNet-B3、DenseNet121、ResNet50+SE、ResNet50+CBAM、weighted CE、label smoothing、focal。
- 核对旧看板进程命令后替换为当前批次，保持13949地址。API核对8组计划、1组已完成、当前variant1。保留旧批次及检查点。当前仅完成启动与批次/GPU检查，未等待整个长时间队列结束。

## 2026-10-29 核对项目验收口径

- 用户询问达到何种结果才算验收。读取当前完整交接包验收说明及整体验证/部位分层协议，核对root预处理目标及优化草案。当前协议规定过程完整性、验证CE选模、冻结后统一阈值总体/模态/部位报告，没有给出正式批准的准确率或AUROC硬门槛；旧预处理目标使用mAP，属于检测历史口径，不适用于当前整图二分类。
- 回复将八组实验完成、研究效果证明与临床有效性区分；建议候选方案多种子复验属于后续建议，未修改当前seed42队列/选模指标或启动额外训练。提醒既有测试集曾有历史评估记录，不能把后续同一集合结果表述成从未查看的独立盲测。未依据历史测试结果调整当前参数或阈值。

## 2026-10-29 八组训练速度再核查

- 用户感觉仍慢，保持训练不变进行只读检查。当前variant1 EfficientNet-B3/none/CE、640、batch4、workers0、缓存v1，进程94248存活；检查时已完成16轮，开始配置文件至最新history约1536.4秒，平均约96.0秒/轮（含首轮预检初始化，非逐阶段profile）。893张train约224批、112张val28批；last.pt约123.2MiB。
- 首次GPU采样96%，随后5次间隔约2秒为24%、99%、36%、99%、51%；总显存约7617-7633/8151MiB（含桌面其他程序），温度70-75℃、功耗约53-57W。CPU消耗约203%-244%（psutil单核100%口径），可用内存约2.2GiB。不能凭这些采样认定唯一瓶颈、显存溢出或热降频。
- 源码仍无AMP、pin_memory或non_blocking，且cudnn.benchmark=False、deterministic=True；每轮验证及保存检查点。缓存移除了反复JPG解码/原尺寸增强，不会移除模型前反向计算。看板只在每轮结束时更新，因此约1.6分钟无新曲线不等于停训。
- 查阅PyTorch官方AMP教程作为下一阶段提速评估依据，建议独立小样本比较AMP与FP32、再考虑异步加载；不承诺倍数。当前正式八组队列未重启、未修改精度/分辨率/batch，避免中途改变比较条件。

## 2026-10-30 记录八组正式结果并准备下一轮优化

- 用户要求记录数据、准备下一轮优化。读取本项目AGENTS、当前批次汇总、八组run_config/history/continue_state、runner及PyCharm配置；进程检查未发现run_architecture_ablation或train_image_classifier。8组状态均complete，完成轮数依次65、27、73、64、65、21、58、75；均在最低CE最佳轮后20轮停止。第0组复用此前同配置seed42基线，不把试跑混入正式结果。
- 新增scripts/record_formal_ablation.py，验证八组身份、共用配置、逐轮连续性、训练893/验证112、阈值0.5、完成边界、汇总指标与最低CE历史一致、数据/缓存/增强/源码指纹一致。运行成功，创建reports/ablation/records/formal_20261030_161652_551258；保存results.json、8行CSV、结果报告、配置/逐轮历史/完成状态/源码快照及inventory。16个best/last权重流式SHA-256记录，权重保留原路径；此记录不是含数据和权重的独立迁移交付包。未对权重做反序列化或推理验证，不能把文件哈希校验称为模型加载验证。
- 独立回读验证49个快照文件与inventory哈希一致、CSV8行、两份新PyCharm XML可解析且数据路径有效；当前数据与缓存manifest哈希符合训练记录，3份训练核心源码均匹配训练时指纹。脚本语法解析成功；git diff --check通过，仅既有文件LF/CRLF提示。控制台中文路径显示编码替代字符，文件通过Python UTF-8读取和XML解析验证，无证据表明文件名或配置内容损坏。
- 最低CE候选为ResNet50+SE+CE（最佳44轮，CE0.3361605547、AUROC0.9447253118、平衡准确率0.8693966970、敏感度0.8550724638、特异度0.8837209302，CM[[38,5],[10,59]]）。保留weighted CE最低CE第1轮全部预测阳性候选的失败结果（CM[[0,43],[0,69]]），不以另一个AUROC最佳轮替换预定选择规则；不声称其所有轮均塌缩。
- 使用experimental-design技能及design_types参考，新增docs/下一轮优化计划_2026-10-30.md：优先补基线/SE的seed43、44，结合既有42形成每方案3种子训练重复；不增加独立验证样本或患者数，不宣称确认性优越结论。seed43按基线→SE，44反序，其他配置保持FP32/640/batch4/原学习率与缓存不变。后续学习率、weighted CE异常检查和AMP吞吐比较列为分阶段待办，未实现或启动AMP。
- 新增.run/optimization_repeat_seed43.run.xml及seed44，调用现有runner分别--variants 0 3 --seed 43和--variants 3 0 --seed 44，无reuse-baseline；依旧为PyCharm运行入口。此次没有启动新训练、切换看板、计算测试性能、修改数据/旧权重或推送GitHub。测试集已有历史评估，同一集合不能再称全新盲测；来源组独立性与临床标签确认仍待解决。

## 2026-10-30 从PyCharm启动复验并切换可视化

- 用户授权启动训练。确认无遗留训练进程后，新增scripts/run_optimization_repeats.py和.run/optimization_repeat_queue.run.xml，将已准备的seed43（0→3）、seed44（3→0）串行执行；任一子队列失败即停止，不并行争用GPU。Python语法与XML解析通过。从PyCharm运行选择器点击Optimization - Four Repeats (seed43 then seed44)，实际launcher104476父进程为PyCharm35128，队列进程101900、首块launcher68320/训练92892，命令包含缓存、workers0、seed43及variants0 3。控制台显示报告reports/ablation/20261030_174612和Starting variant 0。GPU进程列表包含92892，单次利用率97%、总显存5311/8151MiB（含其他应用）。随后用户物理Esc停止界面控制，立即停止Computer Use；没有停止训练。该启动过程补记于本次日志。
- 用户随后要求可视化。再次确认队列与训练进程存活，读取当前配置与历史。扩展training_dashboard.py的--repeat-since模式，将此次起始报告之后首个匹配seed43/44、顺序0 3或3 0的报告聚合成4项，并在HTML显示种子与复验标题，沿用训练/验证损失和AUROC曲线。修正正式模式当前组选择为最新配置创建时间，避免seed44反序执行后仍显示SE。此模式按起始时间和配置匹配，不是通用队列身份登记；不要同时启动另一套同种子同顺序队列。
- 核对13949端口旧看板进程80180及launcher78320的命令后，仅替换看板进程，新launcher82164使用--repeat-since 20261030_174612。训练未重启、未修改数值配置。读取状态验证4项、seed43基线、第1轮、0/4完成；浏览器刷新后可见“基线与SE·多种子复验”、随机种子标识、损失/AUROC曲线与4项进度。截图reports/ablation/20261030_174612/dashboard_repeats.png；看板每3秒刷新，曲线按每轮历史更新，轮内无变化不等于停止。尚未完成后续三次训练，不计算测试性能。

## 2026-10-30 20:47 恢复并展示复验可视化

- 用户再次要求可视化。13949初次连接被拒绝，进程检查未见看板服务，但PyCharm复验队列仍在运行seed44（训练进程106252），未重启或中断训练。
- 重新以隐藏窗口启动只读看板，launcher87940，沿用--repeat-since 20261030_174612及13949端口，使用新时间戳stdout/stderr日志保留旧日志。API恢复成功，显示完成3/4，当前seed44基线已完成第3轮。
- 原浏览器错误页绑定失败，错误原因是连接拒绝页面的data URL；在服务恢复后新建同地址可见标签，实际页面确认多种子标题、3/4及当前曲线。截图dashboard_restored_2047.png保存在该报告目录。未修改训练配置或计算测试性能。

## 2026-10-30 21:48 复验数据归档与三种子报告

- 用户要求记录数据、形成报告。核对20261030_174612和20261030_193451两批汇总/配置/历史/完成状态，未发现训练队列进程；四次新增训练均complete。seed43基线完成33轮/最佳13，SE89/69；seed44 SE77/57，基线54/34。结合seed42两组形成六次记录，均最佳轮后20轮早停。
- 新增scripts/record_repeat_results.py；核验六组身份、连续历史、验证CE选模一致性、样本数、固定阈值、共同配置、数据/缓存/增强/源码/环境一致。流式计算12个best/last权重SHA-256，复制元数据/历史/配置，保留原权重。使用Matplotlib技能生成配对比较PNG/SVG和六面板完整历史PNG，目视检查两图无缺失/截断；统计按3种子均值及ddof1标准差，独立公式校核一致。
- 最终报告reports/ablation/records/repeats_20261030_214844_921038/复验报告.md；results.json保留原始指标、逐轮历史、来源、配置与权重指纹；36文件inventory回读校验通过。初稿repeats_20261030_214756_188506保留；最终版新增训练异常解释后重新生成，作为本次交付版本。
- 均值：基线CE0.4488946/AUROC0.8500169/平衡准确率0.7503090；SE为0.3731166/0.9138299/0.8315920。三个种子CE均改善，提升幅度不同，不能按seed42最高值代表稳定表现。记录seed43基线第26轮CE9.7464、seed44基线第32轮CE2.2515及SE后期波动，不把检查点改进解释为训练全程稳定；具体成因尚未诊断。
- 只归档训练/验证结果，无新训练、测试推理、患者层面统计或Git推送。报告说明候选标签、来源组独立性、重复验证选模偏差及历史测试已查看限制；后续优先小范围学习率对照，AMP单独评估。git diff --check通过，仅LF/CRLF提示。

## 2026-10-30 训练优化总体规划

- 用户要求整理剩余训练日程并形成总体规划文件。读取旧优化构想、启动前复验计划、最终三种子报告和现有代码入口；发现旧计划仍保留待运行历史状态、旧构想含v5与组级标签假设，新规划明确当前v7口径与已完成12次正式缓存训练，保留旧文件。
- 新增docs/模型训练与优化总体规划_2026-10-30.md，列出必要主线P0损失尖峰诊断、P1六段AMP/FP32短测、P2六次SE学习率训练（两新学习率×三种子，复用0.001）、P3冻结、P4评价/解释、P5交付。列明条件扩展及预算、组级/模态创新前提、评价边界、PyCharm执行要求；学习率参数仍待实现，未声称配置已就绪。
- 文件mtime跨度估算四次复验约31.0/77.3/69.0/58.8分钟、52.1–65.4秒每轮；据此6次各100轮估算8.7–10.9小时，另留预检和波动时间。明确非严格profiling、无AMP加速承诺及D1相对日程。报告链接与六项矩阵已核对；本次仅制定规划，未启动新训练或访问测试推理。

## 2026-10-31 识别v8七二一数据包

- 用户要求识别新的7:2:1图片数据集。定位data/raw/口腔整图分类_721完整交接包_2026-10-31_v8，读取包内规则、说明、数据清单及划分元数据；冻结原件未修改，未切换现有训练配置或启动训练。
- 使用包内prepare_721_training_handoff.py的verify只读验证，通过1117张实体图片、1117份XML、2282项根校验和2244项图片目录校验、逐图部位表对应及源码快照校验。来源组、已记录关联簇与相同字节图均无跨集合。报告保存在忽略目录reports/_qa/v8_verify_20261031_180103.json。该校验不是临床标签、患者独立性或模型性能验证。
- 实际划分为train782（292/490、173来源组）、val223（84/139、60来源组）、test112（42/70、35来源组）；约70.01%/19.96%/10.03%。数据仍为原1117张，418正常候选、699病变候选、501荧光、616白光；267划分单元不等于267名独立患者。
- 当前主训练入口--check-only失败：Image inventory differs from manifest。独立枚举确认清单1117张均存在，但旧读取器额外计入1117个._前缀的AppleDouble JPG附属文件，总枚举2234项。此前PowerShell Get-ChildItem未加-Force返回0附属文件，不能作为不存在的依据，已用Path.rglob核实。未删除附属文件或修改冻结包；训练前需在工作代码中显式排除附属文件，并维持真实额外图片的严格清单检查。
- 划分元数据记载501张变更集合，新test含旧train99张、旧val4张及旧test9张。旧训练权重可能已见新test，不得作为独立评价起点；新实验需从约定通用预训练初始化并另建本版缓存，重新划分本身不构成全新独立盲测。旧v7结果不能直接归为本版成绩。

### v8实验初始化约束（用户确认）

- 用户明确要求：若开展本版实验，应从约定的通用预训练初始化重新训练。对当前主要ResNet-50方案，采用ImageNet通用预训练骨干；新增SE及二分类头随机初始化，并记录实际权重标识与随机种子。
- 禁止从旧划分训练产生的best.pt、last.pt或其优化器状态初始化v8新实验；不将旧训练记录接续为v8轮次。仅允许在同一v8数据、配置及预处理版本已核验一致的情况下恢复该次v8实验自身的中断检查点。
- 新实验使用v8清单对应的独立缓存及输出目录，仅以v8验证集选模和确定阈值；测试集留待方案冻结后评价。通用预训练重启可避免继承旧项目权重的测试暴露，但不能消除重新划分和历史数据使用带来的评价限制。
- 本次仅确认并记录约束，未启动训练，未修改训练代码或冻结数据包。

## 2026-10-31 v8三阶段基线实现、启动与实时看板

- 用户要求启动新基线，随后明确严格遵守项目指导书4.3，并选择“训练实时看板：启动训练后显示损失、指标和当前阶段”。取消先前准备沿用的单阶段AdamW方案；第一次UI启动被用户Escape中断时尚未训练，后续在用户选择启动实时看板后继续执行。
- 修改工作代码online_augmentation.py，图片清单检查排除._前缀AppleDouble附属文件，仍拒绝真实额外图片。冻结v8包未修改、附属文件未删除。重新预检通过782/223/112；ImageNet初始化不加载旧项目best/last权重。
- 在data/cache/oral_v8_721_640_v1建立独立缓存：仅train782与val223，不生成test缓存。验证成员、源哈希、缓存哈希和形状；223张验证图片张量与原读取流程逐一bitwise相同。缓存manifest SHA256为78d5f38265ffe3d615ecf26a7f705a7a223f0df0fe5b3941d597e533c72891cc。
- 新增scripts/train_v8_staged_baseline.py及.run/v8_staged_baseline.run.xml。固定ResNet50_Weights.IMAGENET1K_V2，分类头Dropout0.5+Linear2048到2，seed42、640、batch4、workers0、FP32、Adam（L2 weight_decay=1e-4）。第1—20轮只训练fc，lr1e-3；第21—60轮解冻layer3/layer4/fc，lr1e-4；第61轮起全部解冻，lr1e-5。冻结层参数及BatchNorm缓冲统计均冻结。优化器包含所有参数，阶段切换保留Adam状态并更新学习率。
- 不平衡方案选择训练类计数加权CE，权重N/(2*n)：正常候选292张权重1.3390411，病变候选490张权重0.7979591，同时label_smoothing0.1；不叠加SMOTE、欠采样或Focal。本版少数类为正常候选，不能沿用“正常远多于病变”的通用假设。
- 明确实现解释：为满足固定20轮+40轮，早停仅在第三阶段启用，进入第三阶段重新计数；第三阶段无固定轮数上限，普通未加权验证CE连续10轮不改善停止。各阶段全程最低普通验证CE选best，阈值0.5，无测试推理。训练加权平滑损失与验证普通CE定义不同，看板标注不可直接比较。
- 新入口last.pt保存模型、优化器、历史、RNG、第三阶段最优值与计数。续训仅允许同协议/数据/缓存/种子/源码指纹的last.pt；旧数据或旧协议不允许继续。旧image_classification.evaluate_checkpoint新增识别带Dropout的ResNet检查点（未执行测试推理）。旧continue.ps1硬编码v7，不适用于本入口；恢复需使用本入口--resume及原data/cache/seed参数。
- 4项新测试通过：阶段边界、实际冻结权重/BN不更新且分类头更新、优化器状态重载、训练权重+平滑及附属文件/额外图片校验；旧test_ablation.py的3项回归通过。新版check-only通过，git diff --check仅原日志LF/CRLF提示。尚未做跨真实阶段断电重启的端到端训练复验，不将单元测试称为完整恢复验证。
- 新增scripts/staged_dashboard.py及staged_dashboard.html，只读本次v8活动指针与元数据、history、live_progress、continue_state，校验目录及协议并核对PID创建时间。只绑定127.0.0.1:13949，不公开网络；进度约每25批写入，页面3秒刷新。训练和验证损失、AUROC/平衡准确率、灵敏度/特异度、阶段/学习率、可训练参数与第三阶段早停计数均可查看；无数据时不伪造曲线。独立隐藏服务启动，错误日志reports/_qa/v8_dashboard.stderr.log。
- PyCharm经同步文件后显示并点击“V8 - Staged ResNet50 Baseline (Section 4.3)”启动。启动进程117960父进程95016为PyCharm，实际训练进程115608；命令为新入口、v8清单、新缓存及seed42，无--resume。实际运行目录outputs/image_classifier_runs/v8_721_resnet50_baseline_staged_seed42_20261031_181706（相对工作code目录），已确认epoch1/stage1/lr0.001/trainable4098及首批训练输出。
- 已打开并保留http://127.0.0.1:13949/用户可见看板；页面与API确认训练存活、第1轮、批次100/196。启动核查时尚未完成整次训练，不报告最终指标、收敛或后续阶段已执行。未提交或推送Git，旧实验及数据均保留。
- 首次展示后复核：已完成2轮，正在第3轮；前两轮训练+验证约34.8/34.5秒（不含检查点落盘）。第2轮验证普通CE0.6040536、AUROC0.815776、平衡准确率0.774152，均为训练中间值。页面已出现两轮曲线及验证表，浏览器错误日志为空；训练仍运行，无测试结果。

## 2026-10-31 v8同条件架构复用验证

- 用户要求根据既有训练记录用新版数据验证架构复用，随后明确选择“先验证架构：ResNet-50、EfficientNet-B3、DenseNet-121，其他训练条件保持一致”。本轮范围仅三架构，不额外插入SE/CBAM，不开展损失函数对比或多seed；复用网络及实现，不继承旧v7权重。
- 首次看板连接拒绝，进程检查没有训练或看板进程；读取本次v8基线的实际输出，确认已正常完成73轮（20+40+13），stage3_stale10。全阶段最低验证CE在第37轮：CE0.2973649955、AUROC0.9477560808、平衡准确率0.8534172662、灵敏度0.9568345324、特异度0.75；验证223张，阈值0.5。末轮不同于最佳轮，不将末轮性能当作选定模型。训练+验证累计约44.12分钟（不含检查点落盘）。未访问测试推理。
- 核对旧8组匿名结果、基线完整历史/config/state/权重文件；当前源码指纹与基线记录一致。新队列复用已完成v8 ResNet记录需数据/缓存SHA、seed、全部公共协议、历史阶段/学习率/样本数及早停一致，并计算best/last权重哈希。旧v7结果不列入新版比较。
- 新增scripts/v8_architecture_support.py明确各骨干阶段：ResNet layer3/layer4；EfficientNet-B3末端两MBConv阶段features6/7及末端投影8；DenseNet末端denseblock3/4及transition3/norm5。三架构分类头均Dropout0.5+Linear到2，骨干各自ImageNet预训练。EfficientNet内置SE属于架构，不是本轮新增注意力对照。新增test_v8_architectures.py通过三架构×三阶段前/反向、梯度范围、冻结BN缓冲、Dropout及权重重载输出一致检查；合成64像素测试不冒充640正式训练。
- 新增scripts/train_v8_architecture.py，从已验证基线训练入口建立独立版本并替换架构适配，保留原baseline源文件及已完成结果。训练/验证预处理、数据缓存、seed42、Adam/L2、加权平滑损失、20/40轮和第三阶段早停策略保持一致。各架构源码、通用权重枚举与具体解冻范围记录入run_config；新实验不带--resume。
- 新增scripts/run_v8_architecture_queue.py及.run/v8_architecture_queue.run.xml；预检核验基线兼容后依次对B3与DenseNet执行check-only，两组通过且无训练。正式队列先记录已完成基线，再串行启动B3、DenseNet，任一失败则停止队列，保存失败状态。每组完成后校核实际记录并汇总，同时保存配置/历史/状态及源码快照，权重保留原run。新增docs/v8架构复用验证方案_2026-10-31.md，说明固定条件、阶段映射、复用条件及局限。
- 经PyCharm同步外部文件后选择“V8 - Architecture Validation (B3 then DenseNet121)”并启动。最初坐标点击未选中，刷新确认后以Up和Return选中正确配置，没有重复启动旧基线。队列进程117204父进程95016为PyCharm，实际队列121172，实际B3训练78048（经python启动器102760）；报告目录reports/v8_architectures/20261031_200442_656643，B3运行v8_721_efficientnet_b3_staged_seed42_20261031_200451。
- 修改看板以识别两种v8协议及当前架构，显示三组队列结果及对应解冻范围。以新带时间戳日志重新启动本机13949服务，保留原日志；旧浏览器tab不存在，重新创建用户可见tab并保留。页面/API显示1/3完成、B3运行、DenseNet等待，B3阶段1/lr1e-3/可训练参数3074；复核至第4轮，已有真实验证曲线，浏览器错误日志为空。此时未完成整组架构比较，不预报最终排名。
- 新增可选scripts/report_v8_architecture_queue.py用于队列终止后复核并生成文字报告；尝试以Start-Process启动其独立--watch后台进程被自动审批拒绝（仅返回blocked by policy，无具体原因），该进程未启动，未绕过拒绝。训练队列、既有JSON汇总和看板不受影响；后续文字报告需在获准方式下运行，不能声称已自动安排。
- git diff --check仅既有日志LF/CRLF提示。本次未改冻结数据、未提交推送Git、未做测试评价；单seed同验证集比较仅为探索性架构验证，候选标签、来源组及历史测试暴露限制保留。

## 2026-10-31 DenseNet进度文件拒绝访问修复与断点恢复

- 用户提供DenseNet的PermissionError/WinError5及队列CalledProcessError。确认失败发生在第15轮训练结束后、验证前更新live_progress.json的os.replace；原队列失败停止，无遗留训练。文件非只读，未证实具体占用进程；不能将Windows临时文件共享冲突的推断写成已定位看板或杀毒软件。
- 同版last.pt可读取，完整保存第14轮、优化器状态、14轮历史及Python/NumPy/Torch/CUDA/DataLoader RNG，数值训练源码指纹与当前完全一致。第15轮尚无完成检查点，因此从第14轮边界恢复并重跑第15轮。ResNet73轮及EfficientNet78轮已完成结果保留；B3选定第55轮，验证CE0.2814416831/AUROC0.9605173004/平衡准确率0.8949554642，仅验证集探索结果。
- 新增scripts/resilient_training_io.py：每次独立临时文件、替换遇PermissionError最多10次有界退避重试，替换前设置文件权限；仅live_progress.json持续I/O失败警告并跳过，历史、状态、检查点等关键写入失败继续抛错。仅清理本次创建的临时文件，不删除历史遗留.tmp或重要原件。
- 为保留历史训练源码与严格指纹检查，未改train_v8_architecture.py、image_classification.py或网络/损失/策略实现。新增resume_v8_with_resilient_io.py先核验原数值源码指纹，再显式替换该进程的JSON及检查点I/O函数；io_adapter.json及后续检查点io_adapter字段记录适配器/包装器SHA256，不伪装成无修改环境。原identity继续标识实际保持不变的数值训练源码。
- 新增tests/test_resilient_training_io.py，4项通过：瞬时占用重试、持续进度失败可跳过而关键文件失败必须报错、检查点读回及Torch RNG不变、真实Windows CreateFileW不共享删除的读句柄阻塞原子替换后释放成功。该测试验证文件机制，不证明已识别本次实际占用者。
- 新增recover_v8_architecture_queue.py及PyCharm配置“V8 - Recover DenseNet121 (I-O retry)”。恢复前重新核验两组已完成记录，归档失败队列JSON、DenseNet第14轮best/last、history/config/state/live及遗留tmp和修复源码至reports/v8_architectures/20261031_200442_656643/io_recovery_20261031_212406_567047。原失败记录留档，当前队列新增recovery_attempts，不重跑已完成模型。
- PyCharm最小化导致首次CUA取状态失败；通过已核验窗口激活后同步配置，选择恢复入口启动。启动进程124916父进程95016为PyCharm，实际恢复协调96376、训练120748（经84804启动器），命令显式含同版DenseNet last.pt、原v8数据/缓存/seed42。恢复协调在完成后核验DenseNet结果并更新原队列，文字报告由其前台子进程生成，不另启被拒绝的后台watcher。
- 恢复后核验：第15轮已完成223张验证，last.pt更新至epoch15，优化器状态及io_adapter记录均可读，看板进入第16轮。归档中断前第15轮train_loss与恢复重放后的history均为0.6230373645530027，数值完全一致；该证据支持本轮训练损失重放一致，不推断与未知不中断运行的全部模型张量逐位相同。训练继续，未做测试集评价或Git提交。

## 2026-10-31 v8架构结果归档与候选选择

- 三架构均已完成并重新核验完整阶段、早停、数据/缓存及权重哈希；ResNet73轮最佳37，B3共78轮最佳55，DenseNet80轮最佳39。
- 按预定最低验证普通CE规则选DenseNet-121，CE0.2434586092、AUROC0.9610311751、平衡准确率0.9032631038、灵敏度0.9136690647、特异度0.8928571429；使用第39轮best.pt。单seed探索性选择，未宣称稳定显著优于B3。
- 完整决策、指标CSV及可复用权重清单：reports/v8_architectures/20261031_200442_656643/selection_20261031_221746_729709。建议下一阶段先对DenseNet/B3配对seed复验，再单因素消融；本次没有启动新增训练、测试集评价或Git提交。
