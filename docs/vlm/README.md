# VLM 主线：视觉理解（图像/视频 → 文字）

让模型把视觉世界映射到语义/语言空间，是世界模型的"感知与接地"入口。

## 主题清单

- [x] [`vit-basics.md`](vit-basics.md) —— 视觉编码器：ViT、CLIP、SigLIP，如何把图像变成 token
  <br>结论：ViT 只在"切 patch → 线性投影 → token"这一步创新，之后照搬标准 encoder；patch 的语义不是切出来就有的，是训练信号塑造的。监督分类把语义锁在固定类别里，CLIP/SigLIP 的图文对比学习才把图像和文本压进同一个可做 attention 的空间。
- [ ] `alignment.md` —— 模态对齐：如何把视觉特征接到 LLM（projector/Q-Former/cross-attn）
- [ ] `llava-line.md` —— LLaVA 系：极简 projector + 指令微调的范式
- [ ] `qwen-internvl.md` —— Qwen-VL / InternVL：动态分辨率、原生多模态的演进
- [ ] `training-recipe.md` —— 训练配方：预训练对齐 + 指令微调 + 数据构造
- [ ] `eval.md` —— 评测：MMMU、MMBench、幻觉与 grounding 的度量
- [ ] `video-understanding.md` —— 视频理解：帧采样、时序建模

## 奠基论文

- ViT (Dosovitskiy et al. 2021)、CLIP (Radford et al. 2021)、SigLIP (2023)
- BLIP-2 / Q-Former (Li et al. 2023)
- **LLaVA** (Liu et al. 2023) 及 LLaVA-1.5
- Qwen-VL / Qwen2-VL、InternVL 系列技术报告

> 完成一个主题后把 `[ ]` 改成 `[x]` 并留一句话结论。
