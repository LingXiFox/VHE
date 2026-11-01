# 数据授权、隐私与泄漏防护

候选训练语料：AISHELL-3（官方 OpenSLR 93，中文文本和拼音，Apache-2.0）。实际获取前重新核对来源、授权与附带 README。大型数据**不自动下载**。

允许：在相应许可范围内为非商业研究本地处理、分析、做派生训练对。外部发布音频样本前核对许可条款与署名要求。
禁止：提交原始数据到 Git；共享带个人身份/非授权音频的样本；用未经授权的音色克隆数据做项目展示。不要把训练集中的真实说话人称为原创音色。

路径建议：`data/raw/aishell3/`、`data/processed/`，均在 .gitignore；只提交 `data/manifests/*.jsonl` 的脱敏字段（文件相对路径或匿名 ID，SHA256、划分，无个人身份）。

拆分策略：以说话人 ID 为单位分离训练/验证/测试；再对模板随机种子做独立 holdout；训练中使用的任何音频片段、派生文件不得出现在最终评估集中。

外部代码与模型：记录来源、版本、具体许可证，禁止直接把不兼容许可内容合并到仓库。

## 已核实的 AISHELL-3 事实（2026-10-31，curl 直连官方页与镜像）

- 官方页 https://openslr.org/93/（SLR93），论文 arXiv:2010.11567，许可 **Apache License v.2.0**。
- 发布包 `data_aishell3.tgz` 约 **19 GB**；85 小时、218 说话人、88035 条录音。
- 根目录：`spk-info.txt`（`说话人<TAB>年龄段<TAB>性别<TAB>口音`，前两行为 `#` 注释）、`phone_set.txt`。
- `train/content.txt` 形如 `SSB00050001.wav<TAB>广 guang3 州 zhou1 女 nv3 ...`（汉字与带声调数字拼音交替）；
  `train/label_train-set.txt` 是可选韵律标注；`test/content.txt` 同格式，但 test 没有 label 文件。
- 音频为 **44100 Hz / mono / 16-bit PCM**，单条约 9.5 s。契约只支持 16000/24000/48000 Hz，
  **配对时必须显式重采样并把重采样写进记录**，不得悄悄改变采样率。
- 唯一适配实现：`src/vhe/data/aishell3.py`；`python -m vhe manifest` 产出 JSONL（meta + utterance + issue），拒绝覆盖已有文件。
- manifest 只保存相对路径、SHA256、划分与说话人属性，不含音频；真实语料不入库。

## 说话人划分与派生数据（2026-11-01 实测）

- **官方 train/test 是按句切分的**：`test/` 的 214 个说话人目录里 170 个与 `train/` 重名，并集恰为 218。因此官方 test 含说话人泄漏，不能用于 speaker-independent 评估。
- 本项目改用说话人划分 **180 / 19 / 19**（seed 20261101，说话人 ID 排序后洗牌），原始音频路径不变；同一说话人的全部句子只落在一个 split（`tests/test_data_corpus.py` 有断言）。
- 派生数据不进 Git，统一放数据集同盘：
  - `/Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v2/manifest.jsonl`（sha256 `651bbbac…`，88035 行）与 `report.json`；
  - `derived/aishell3-scan/v3/manifest.jsonl`（sha256 `10539f61d015bc81dff88cb0ee5ba72b640ab703ee57373324f7c446a624c80f`，88035 行，manifest v2 格式）与 `coverage_p14_baseline.json`：**当前用于 P1.4 的基线**，2 句带 `prosody_conflict` 质量标记；v2 及其 `report.json` 保留不删（v3 只是加了质量标记，条目内容不变）。
  - `derived/aishell3-audit/v2/prosody_audit.json`：四分类审计（checked 63262 / matched 63260 / mismatched 2 / format_errors 0 / missing_text 0 / header_comments 4），v1 保留不删。
  - `derived/aishell3-scan/v1/`：**有缺陷的首次扫描，保留作证据**，不得删除或覆盖；
  - `derived/aishell3-audit/v1/prosody_audit.json`：韵律标注审计结果。
- manifest 体积（数万行、含逐句音素）不进 Git，只提交生成命令与 sha256；复现靠 `python -m vhe scan`。
- 文本以 `content.txt` 为准；`label_train-set.txt` 仅做韵律交叉核验。已确认 2 处标注冲突（`SSB19560288` `ni2`/`ni3`、`SSB15550353` `zheng1`/`zeng1`）；原记「4 条未标注」经复核为 `label_train-set.txt` 第 1–4 行的 `#` 文档注释（EXP-0013 末尾已追加勘误，历史结论未改）。配对时必须打 `prosody_conflict` 标记，禁止静默择一（v3 manifest 已落盘该标记）。

## 覆盖率复算规则（2026-11-01，EXP-0016）

- 扩表后的覆盖率一律用 `stats_from_entries(manifest 条目)` 重算，**不重解码任何 WAV、不重建索引**：统计只依赖文本与音素表，重扫 26 GB 只会引入新的失败面。
- 每个批次写新的 `coverage_stage{N}.json` 与 `phonemes_stage{N}*.json`，**不覆盖旧报告**；被判定为度量缺陷的版本（如 `phonemes_stage2_v3.json`）原样保留，结论以 README 指定的定稿版为准。
- 覆盖率上升**不等于**听感改善，也不能用来宣称通过 P1 验收：整句覆盖率是音节级的乘积式限制，任何听感结论都需要人工听音记录。
