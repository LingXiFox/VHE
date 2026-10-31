# EXP-0004 — AISHELL-3 适配器：目录发现、失败分类与按说话人分组的 manifest

## 假设
给定本地 AISHELL-3 目录树，仅依赖 `content.txt` 与 WAV 文件头构建 manifest，可以做到：
每一条被接受的录音都带稳定的 SHA256 溯源信息；每一个被拒绝的文件都产生可计数的 issue code；
按说话人分组切出的 dev 不破坏官方 test 划分，且可证明没有说话人跨 split 泄漏。
判据：适配器单测全绿、CLI 在同一输入上产出字节可复现的 JSONL 且拒绝覆盖、4 类坏样本各计 1 次。

## 变量 / 对照组 / 数据划分
- 唯一变量：数据适配层（解析 + 校验 + 分组 + manifest I/O）。不涉及模型、不涉及训练。
- 对照组（负样本，全部来自合成夹具）：`missing_wav` / `not_mono`（2 声道）/ `empty_text`（只有 TAB 没有转写）/ `too_short`（0.1 s < 0.5 s）。没有适配层时这四类会被静默混入训练集，因此“各计一次且不进 entries”就是这一层的验收点。
- 附加样本：`pinyin_mismatch`（转写 4 字、拼音 3 音节）必须只记为 warn 并保留条目。
- 数据：**本机没有真实 AISHELL-3**（见“失败与阻塞”），全部测试使用 44100 Hz / 16-bit / mono 合成夹具。目录结构、`content.txt` 首行、`spk-info.txt` 首行均逐字复制真实语料（已通过 curl 核实，见 docs/DATA_POLICY.md）。
- 划分：官方 test 说话人保持不动；dev 从官方 train 说话人中按 `seed` 确定性抽取；说话人是划分的最小单位。

## 完整复现命令 / 环境与哈希
```bash
.venv/bin/python -m pytest tests/test_data_aishell3.py -q          # logs/pytest_adapter.txt
.venv/bin/python -m pytest -q                                     # logs/pytest_full.txt
.venv/bin/python -m vhe manifest --root /tmp/vhe_exp0004/AISHELL-3 \
    --output experiments/EXP-0004/logs/manifest_demo.jsonl --dev-fraction 0.5   # logs/cli_manifest.json
shasum -a 256 experiments/EXP-0004/logs/manifest_demo.jsonl        # logs/manifest_sha256.txt
```
环境：macOS 27.2 (26B123) / arm64 Apple M5 / Python 3.14.6 / torch 2.15.0（MPS 可用）。
Git：`3a9149d` + 未提交改动（`M src/vhe/cli.py`、`?? src/vhe/data/`、`?? tests/test_data_aishell3.py`）。
seed：`--dev-fraction 0.5 --seed 42`（manifest_demo）；夹具内 wav 由 `vhe.audio.write_wav` 确定性生成。

## 预注册通过/失败条件
1. `tests/test_data_aishell3.py` 全绿（14 项）。
2. 全库 `pytest -q` 全绿，唯一跳过项必须是“需要真实 CUDA”。
3. 同一输入两次生成的 manifest 字节相同、SHA256 相同；已存在的 manifest 必须被拒绝覆盖。
4. 四类坏样本在 `stats.error_codes` 中各自恰好为 1，且 `entries` 不含它们。
5. 任何说话人不得同时出现在两个 split（`check_no_speaker_leak` 不抛错且分组表符合预期）。

## 实际结果与主观听音
| 项 | 结果 |
| --- | --- |
| 适配器单测 | **14 passed**（`logs/pytest_adapter.txt`） |
| 全库 pytest | **49 passed, 1 skipped**（跳过 = `tests/test_model.py:35` 需要真实 CUDA） |
| 坏样本计数 | not_mono 1 / empty_text 1 / missing_wav 1 / too_short 1；warn: pinyin_mismatch 1 |
| entries | 4 条（train 2、test 1，dev 由 0.5 比例抽出 1 个说话人） |
| manifest SHA256 | `d49953ec4d33bc0d1a1781d20a35f85e6c3f911563be2efa91af3e015a612cde`（10 行 = 1 meta + 4 utterance + 5 issue） |
| content SHA256 | `744fe02daeed62ab5b204883063f4415a457deabc0b6a689040bf8ce0cb78fce` |
| 覆盖保护 | 第二次运行退出码 2：`Manifest build failed: Refusing to overwrite manifest: ...` |
| 说话人泄漏 | `check_no_speaker_leak` 通过；split 表 train/dev/test 各 1 个说话人 |

本实验**没有**任何主观听音结论：输入是合成正弦音，不是真人录音；也不存在任何“自然度”声明。

## 失败、负面发现、错误日志
1. **流程偏差（必须记录）**：适配器代码与测试先于本记录创建（与 EXP-0003 相同的违反：AGENTS.md 要求运行前建记录）。本记录只覆盖验证运行；没有回溯修改历史。
2. 首轮 4 项测试失败，均已修复并保留在历史里：
   - `empty_text` 被 `line.strip("\ufeff \t\r\n")` 吃掉了尾部 TAB，退化为 `bad_line` —— 真实缺陷：会掩盖真实语料里的空转写。修法：只 `rstrip("\r\n")` + `lstrip("\ufeff")` 后再判断 TAB。
   - 期望 test 说话人含 `SSB0012`，但其唯一音频是 2 声道，被 `not_mono` 拒绝 —— 期望错误，不是代码错误；已改为只含 `SSB0011`。
   - 重写夹具 wav 触发 `write_wav` 的拒绝覆盖 —— 夹具需先 `unlink`。
   - 测试路径漏了说话人目录（`train/wav/SSB00050001.wav`），写到了不存在的路径，导致“哈希不变”的假失败。
3. **真实数据阻塞**：本机 `data/raw` 不存在，`~/data`、`~/Datasets`、`~/Downloads`、`/Volumes/Development/Projects/datasets` 与 Spotlight 均无 AISHELL-3；未获授权不得下载 19 GB 语料。因此真实语料上的解析率、时长分布、听音抽检全部未做，P1 的真实数据验收**未完成**。
4. `web_search` 工具余额不足（HTTP 402）、`web_fetch` 被沙箱拒绝（non-public IP），格式核实只能通过 `curl` 完成；这条限制记录在此，避免后续误判信息来源。

## 结论 / 下次实验
适配层在夹具上可复现、失败可见、划分不泄漏；真实语料验收被数据缺失阻塞，不能宣称 P1 完成。
下一步（P1-B）：在重采样之前先做 F0 提取与音节切分，把 44.1 kHz 原生语料降采样到契约支持的 24 kHz，并把每一次重采样与 F0 失败写进 manifest issue，而不是静默过滤。
