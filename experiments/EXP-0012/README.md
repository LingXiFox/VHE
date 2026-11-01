# EXP-0012 — AISHELL-3 corpus index, speaker-disjoint split and 26-phoneme coverage

## 假设

The published train/test split is utterance-level, not speaker-level (170 of 174 train speakers also appear under test/), so a speaker-disjoint 180/19/19 split is required before any speaker-independent claim; with the current 26-phoneme inventory fewer than half of the syllable tokens are renderable, and a small set of missing onsets (zh, ch, h, r) plus frequent finals accounts for most of the blocked mass.

## 变量 / 对照组 / 数据划分

- 唯一变量：划分方式。对照组 = 官方 `train/` + `test/` 目录（按句切分）；实验组 = 按说话人划分 180 / 19 / 19（seed 20261101）。
- 不变量：26 音素 inventory（`mandarin`）、`missing_units` 判据、`content.txt` 文本。
- 数据：`/Volumes/LingXiWorkspace/VHE-Datasets/AISHELL-3/extracted`，88035 条，218 说话人，85.62 h。
- 分组：说话人为最小单位，同一说话人的全部句子只出现在一个 split（有测试断言）。

## 完整复现命令 / 环境与哈希

```
python -m vhe scan \
  --root /Volumes/LingXiWorkspace/VHE-Datasets/AISHELL-3/extracted \
  --manifest /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v2/manifest.jsonl \
  --report   /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v2/report.json \
  --top 12
```

- manifest sha256：`651bbbac1468305c47b8fee2be417561a152097cca0edc5dba5313ef2c1e0093`
- 环境：macOS arm64 / Python 3.14.6 / CPU（无 torch 参与）；Git `43d3218` + 未提交工作树（见 metadata.json）。
- 扫描只读 `content.txt` 与 WAV 头（44 字节），不解码波形。

## 预注册通过/失败条件

1. 说话人零泄漏（断言）→ 通过。
2. 解析错误 = 0 → 通过（v2）。
3. 音节覆盖率 < 50%（假设）→ 通过，实测 30.95%。
4. 失败线：任一 split 说话人 < 10 或出现跨 split 句子即判失败。

## 实际结果与主观听音

- 索引：**88035 句 / 218 说话人 / 0 错误 / 997,997 音节 / 85.6178 h**；音节总数与 `phone_set.txt` 词频总和完全一致，说明音节切分无遗漏无重复。
- 划分：train 72966 句 / 180 人，dev 8189 句 / 19 人，test 6880 句 / 19 人。官方 `test/` 的 214 个说话人目录中有 **170 个与 train 重名**（并集恰 218），确认官方划分按句而非按人。
- 覆盖率：音节级 **0.309544**（308,924 / 997,997 可渲染），阻塞 689,073；句子级仅 **187 / 88035 = 0.0021** 整句可渲染。第二项远低于预期，是本实验最重要的负面结果。
- 声调：1:210317 / 2:235098 / 3:160859 / 4:330936 / 5:60787。
- 声母（top，含 y/w 正字法滑音与零声母 `-`）：y 97143, d 93131, sh 80440, j 71162, l 54043, zh 53341, x 51245, b 49426, g 46985, w 43313, h 39711。
- 韵母（top）：i 91385, e 91073, u 61615, apical_retroflex 53935, ian 47442, ai 41550, uo 38153, ing 37436, en 36857, iou 34896。
- 阻塞质量（top）：final:e 91073, final:apical_retroflex 53935, onset:zh 53341, final:ian 47442, onset:h 39711, final:uo 38153, final:ing 37436, final:en 36857, final:iou 34896, final:ong 32588。
- 下一批音素（按"加入后能整音节解锁多少句"的精确边际排序，非单点频次）：final:e 74797 → final:ian 47442（累计 122239）→ final:apical_retroflex 37864（160103）→ final:ing 37436（197539）→ final:iou 34896（232435）→ final:uo 34783（267218）→ onset:zh 34606（301824）→ final:ong 29562（331386）→ final:en 26197（357583）→ onset:h 25585（383168）。
- 未做听音，本实验不涉及主观评价。

## 失败、负面发现、错误日志

- **v1 扫描（保留为缺陷证据）**：`/Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v1/`，manifest sha256 `9ed0cc56…`，78842 句成功 / **9193 句 `bad_pinyin`**。根因：`mandarin.resolve_final` 的 j/q/x 分支提前 return，把 `jiu/qiu/xiu` 的缩写 `iu` 当韵母查表失败（错误文本 `'jiu3': j + 'iu' is not a Mandarin final`）；第二处是 `missing_units` 把 y/w 当声母，虚增 `onset:y` 86439、`onset:w` 37524。v1 报告未删，用于记录"覆盖率 0.2728 是被解析缺陷低估的"。
- 修复后 v2 才 0 错误（`src/vhe/data/mandarin.py`：`"iu": "iou"`；y/w 不计声母缺口）。
- 未解决：`utterance_coverage = 0.21%` 意味着按整句成对构造时几乎全部样本被阻塞，P1.4 必须先扩 inventory，不能靠跳过阻塞句凑数据。

## 结论 / 下次实验

- 官方划分不能支撑 speaker-independent 声明；180/19/19 划分可用且零泄漏；26 音素 inventory 的可用度远低于预期。
- 扩展顺序：韵母 e / ian / apical_retroflex / ing / iou / uo / ong / en / uei / v + 声母 zh / h / ch（zh/ch/h/r 与舌尖韵母互相牵连，见 `also_needs`）。
- 下次：EXP-0015 按上述边际顺序扩 inventory 并重测；同时修 F0 估计器（EXP-0014 的八度错误），否则配对数据 F0 不可信。
