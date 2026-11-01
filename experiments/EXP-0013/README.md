# EXP-0013 — Pinyin orthography vs actual pronunciation in AISHELL-3 annotations

## 假设

A letter-level pinyin-to-phoneme mapping mislabels a measurable share of syllable tokens: y/w spellings hide i/u/ü medials, j/q/x + u is ü, iou/uei/uen are abbreviated to iu/ui/un, apical i after z/c/s/zh/ch/sh/r is not /i/, labial + o is /uo/, and 儿化 is written as a trailing r; an explicit orthography-resolution layer can parse every one of the 1920 distinct corpus tokens with zero silent substitutions.

## 变量 / 对照组 / 数据划分

- 唯一变量：拼音解析层。对照组 = 按字母直查韵母表（前一轮 `PINYIN_FINAL_TO_PHONEME`，8 项）；实验组 = `mandarin.split_token` 正字法解析（缩写还原、ü 变体、舌尖元音、y/w 滑音、儿化）。
- 数据：`phone_set.txt` 的 3393 行 token（去重 1920）+ `content.txt` 全量 997,997 个音节 token。
- 标注一致性：`label_train-set.txt`（韵律标注）与 `content.txt` 逐 token 对比，只做交叉核验，不用标注覆盖文本。
- 判据：distinct token 解析失败数 = 0；且不得出现"近似替换"（任何未在音素表中的目标必须报阻塞，而不是猜一个相近音素）。

## 完整复现命令 / 环境与哈希

```
python -m vhe audit \
  --root /Volumes/LingXiWorkspace/VHE-Datasets/AISHELL-3/extracted \
  --output /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-audit/v1/prosody_audit.json
python -m vhe scan --root ... --manifest .../v2/manifest.jsonl --report .../v2/report.json
```

- Git `43d3218` + 未提交工作树；环境 macOS arm64 / Python 3.14.6。
- 审计对象：`train/label_train-set.txt` 的 63262 行。

## 预注册通过/失败条件

1. 1920 distinct token 解析失败 0 → 通过（初版因 `iu` 缺陷失败，修复后通过，见下）。
2. 全量扫描 `bad_pinyin` = 0 → 通过（v2）。
3. 标注不一致率 < 0.1% → 通过，实测 2 / 63262 = 0.0032%；但这些不一致必须逐条列出，不允许静默取其一。

## 实际结果与主观听音

- 正字法覆盖：`phone_set.txt` 3393 行 / **1920 distinct token 全部解析成功（0 失败）**；全量 997,997 音节 **0 个 `bad_pinyin`**。
- 规则被实际触发的证据（出现在 `next_units`/统计中的单元）：缩写韵母 `iou` 34896、`uei` 27950、`uen`（含于 `zh/ch/sh/r + uen` 组合）；舌尖韵母 `apical_retroflex` 53935、`apical_dental` 20702；ü 韵母 `v` 23661；j/q/x + u 判定为 ü 的分支。
- 标注审计（EXP-0013 输出 `…/aishell3-audit/v1/prosody_audit.json`）：checked **63262**，matched **63260**，**mismatched 2**，unlabelled 4。
  - `SSB19560288`（label 第 37163 行）index 4：label `ni2` vs content `ni3`。
  - `SSB15550353`（label 第 56267 行）index 2：label `zheng1` vs content `zeng1`。
- 未做听音，不宣称发音正确性。

## 失败、负面发现、错误日志

- **初版失败**：用直查表解析时 9193 句 `bad_pinyin`，典型错误 `'jiu3': j + 'iu' is not a Mandarin final`——j/q/x 分支提前 return 导致 `iu` 缩写未还原。这是 EXP-0012 v1 扫描数据不可用的同一根因，此处再次记录，不修改历史结论。
- **第二类错误**（非解析失败但统计错误）：`missing_units` 原先把 `y`/`w` 当作声母，产生 `onset:y` 86439 / `onset:w` 37524 的虚假缺口；y/w 是正字法滑音（`ya` = `ia`），不是独立声母。已修正并加测试。
- 未解决：`unlabelled 4` 的具体 utt_id 目前只在计数里，未逐条落盘；`prosody_audit.json` 未保存这 4 条的标识，下一次审计需要补字段。两处标注冲突（`ni2/ni3`、`zheng1/zeng1`）尚未在成对数据里加"标注冲突"标记位。

## 结论 / 下次实验

- 正字法解析层可用：0 失败、0 静默替换，缩写/ü/舌尖元音/儿化全部按规则还原。
- 处理政策（写进 DATA_POLICY）：文本以 `content.txt` 为准；`label_train-set.txt` 只用于韵律交叉核验；两处冲突样本在配对时打 `prosody_conflict` 标记，绝不静默择一。
- 下次：EXP-0015 扩 inventory（需要新增的正是 `apical_retroflex/dental`、`v`、`iou/uei/uen` 这些已能正确识别的单元），并补审计的未标注 ID 落盘。
