# 相关研究（初始检索，非新颖性结论）

1. Tanaka et al., **WaveCycleGAN: Synthetic-to-natural speech waveform conversion using cycle-consistent adversarial networks** (2018): https://arxiv.org/abs/1809.10288 。重要先例：合成声音波形 → 自然化波形，不能声称此任务从未被研究。
2. Tanaka et al., **WaveCycleGAN2: Time-domain Neural Post-filter for Speech Waveform Generation** (2019): https://arxiv.org/abs/1904.02892 。时域神经后处理。
3. Caillon & Esling, **RAVE: A variational autoencoder for fast and high-quality neural audio synthesis** (2021): https://arxiv.org/abs/2111.05011 。音频潜变量表示和两阶段训练。
4. **AISHELL-3: A Multi-Speaker Mandarin TTS Corpus** (2020/2021): https://www.openslr.org/93/ 。中文、多说话人、汉字和拼音标注；数据许可 Apache 2.0。
5. PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html 。跨平台、跨版本不保证完全可复现；需要记录精确环境与 seeds。

文献检索待办：查 speech enhancement、neural vocoder post-filter、conditional GAN voice conversion、speaker-independent speech naturalization、articulatory synthesizer inversion + neural refinement、manner/prosody preservation。

**新颖性审慎规则**：当前未发现公开项目完整满足「无需 VHE 外部条件 + 任意未见原创音色 + 保留身份 + 自然化」并不等于没有。论文前必须系统检索并引用最近成果。
