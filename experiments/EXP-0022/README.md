# EXP-0022 — Per-state transition re-estimation for the chain-HMM aligner

## 假设
Fixed 0.5 stay probabilities force every phoneme class to an expected 60 ms (3 states x 2 frames each), which EXP-0021 measured as a vowel/stop duration ratio of 1.43. Re-estimating one stay probability per state from expected transition counts should let decoded phone durations follow the corpus instead of the prior, without changing anything else.

## 变量 / 对照组 / 数据划分
- 唯一变量：`estimate_transitions`。False = 固定自环 0.5（复现 EXP-0021）；True = 每轮 M 步用期望转移计数重估**每个状态**的自环概率，Beta(2,2) 平滑，clamp [0.05, 0.98]，未被触及的状态保留原值。
- 固定不变：manifest `1a8138c4d13e547d1d266a7aaf50d3a002a8e32dad78b0555f1b417c77074f31`、16 个训练说话人 × 20 句、4 个 dev 说话人 × 40 句评估、3 states/phone、1 高斯分量、seed 42、frame_cap 2500、4 轮 EM。
- 说话人划分沿用既有 split（train/dev 说话人不重叠，与 EXP-0021 完全相同的 `select_utterances` 抽样），因此两实验的逐句评估集一致，可直接对比。

## 完整复现命令 / 环境与哈希
见 metadata.json；两臂命令只差一个开关：

```bash
PYTHONPATH=~/VHE/src .venv/bin/python experiments/EXP-0022/probe_corpus_aligner.py \
  --manifest ... --root ... --train-speakers 16 --eval-speakers 4 --per-speaker 20 \
  --eval-utterances 40 --iterations 0 4 --seed 42 \
  [--estimate-transitions] --out logs/...json
```

## 预注册通过/失败条件
运行前写下，事后不改：
1. **主判据**：解码音素时长按类聚合的 vowel/stop 平均比 ≥ 2.0（EXP-0021 固定转移为 1.43）。
2. 地板比例（平均时长 ≤ 10 ms 的音素占比）≤ 5%（EXP-0021 为 5.2%）。
3. 平均边界置信度 ≥ 0.80（EXP-0021 为 0.816）。
4. 训练似然逐轮单调不降。
5. 回归护栏：与能量谷 DP 的 50 ms 内一致率 ≥ 25.7%（EXP-0021 的 30.7% 减 5 个百分点）。DP 参照物本身未人工验证，只作护栏，不作正确性证据。
判据 1 失败即假设被证伪，需要另找时长塌缩的原因（发射模型容量、帧长、参照物）。

## 运行前的合成信号预实验（本机 mac，已记录，事后不改）
- 12 条 2 音素合成句（"a" 300 ms + "t" 60 ms，频带互不重叠），flat start + 3 轮 EM。
- 两臂解码时长**完全相同**（a≈0.215 s、t≈0.157 s）：估计臂只改变模型隐含时长（a 0.181 / t 0.155），没有移动 Viterbi 边界。该合成集上发射项主导，判据 1 因此有失败风险。
- 预实验中查出并修复的结构性缺陷：链必须在末态结束，末态的 arc 转移计数结构性为 0，导致末态自环概率饱和到 0.98、隐含时长虚高（"t" 报 0.42 s，而解码只有 0.157 s）。修复方式是把"终止"记为该末态的一次退出事件；修复后隐含时长与解码时长一致。
- 这说明"模型隐含时长"本身在修复前不可作为判据 1 的旁证，故两臂的类时长一律以**解码 spans** 统计。

## 实际结果与主观听音
未运行。

## 失败、负面发现、错误日志
TODO（即便未成功也必须记录）

## 结论 / 下次实验
TODO
