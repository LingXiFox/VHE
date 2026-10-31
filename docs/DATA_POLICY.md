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
