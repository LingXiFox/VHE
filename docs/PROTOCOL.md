# 实验协议 v0.1

## 样本与时间对齐

真人样本 `(audio, text, pinyin, speaker_id)` → 音素分段、声调/韵律、F0、能量 → 在训练数据制造环节生成 `mechanical_audio`。每对样本有 `pair_id`, `speaker_id`, `source_sha256`, `mechanical_sha256`, `alignment_quality`, `synth_version`, `config`, `split`。

VHE 推理严格只有机械波形 `[B,1,N]`；处理失败时抛错并记录。模型可以在内部计算 Mel/STFT，但不能接触构对阶段的文本、说话人身份、真人音频。

## 分阶段计划与对照

- P0：确定性机械生成与记录工具，零神经训练。
- P1：少量授权数据，构造高质量配对，确认音素、F0、时长匹配；打印失败样本；禁止先用对齐误差污染训练结论。
- P2：小型 VHE 残差 Conv1d/1D U-Net，baseline 输入/输出形状和长度完全一致，初始残差为零。冻结上游。
- P3：真实重建/对齐目标、STFT + Mel 重建，严格衡量在真实目标上重建是否引发音色替换；额外测试随机原创模板。
- P4：针对未见模板加约束；逐步引入真实感判别器、feature matching 或 VAE 潜变量。每项单独消融，禁止多项一起堆砌而无法归因。

## 指标（每项同时报告均值和分布）

1. 自然度：随机双盲 A/B 偏好，或 MOS；报告受试人数、有效样本数、统计区间、耳机/扬声器、刺激音量一致化方式。
2. 内容：中文 ASR CER（合成/自然的识别性能存在域偏差，辅助人工听音），特别是辅音、韵母、声调。
3. F0：voiced frame 的均值/分位偏差、outlier ratio；记录无声段分割失败率。
4. 时序：长度、音素边界偏差、停顿变化、波形裁剪或拖延。
5. 音色保持：同一输入模板在多句上的身份一致性、参考听音分组；speaker embedding 只作为有域偏差的辅助指标。
6. 频谱与质量：Multi-resolution STFT/Mel、能量/LUFS、削波率、噪声；不把频谱距离当自然度的代名词。
7. 泛化：speaker holdout + template holdout；必要时 synth engine holdout（不同合成器）。

## 常见致命混淆

- 不要随机按 utterance 划分，会把同一人的音色泄漏到测试集。
- 不能用训练时的自然人声作为推理侧的隐藏参考。
- 真实声音与机械音色必须足够对应，配对时的波形距离否则会鼓励网络做变声。
- 训练集上的重建质量不是原创模板泛化能力。
- F0 不能一概等于 MIDI note；中文语音的韵律来自实际发声，唱歌另行测试。
- 任何自动对齐/基频提取需要标注不可信结果的比例，不能把失败悄悄丢掉。

## 最低可复现字段

experiment_id, timestamp_utc, git_commit, git_dirty, python_version, OS, GPU, CUDA_driver, torch_version, dependency_lock, seeds, dataset_name/version/license, manifest_hash, train/val/test_split_hash, synthesis_version, config_sha256, command, runtime, peak_vram, metrics, artifacts_sha256, failure_notes.
