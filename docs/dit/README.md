# DiT 主线：生成与模拟（文字/条件 → 图像/视频）

离世界模型最近的一条线。建议按"数学基础 → 架构 → 视频 → 前沿"的顺序推进。

## 主题清单

- [x] [`diffusion-basics.md`](diffusion-basics.md) —— 扩散模型基础：前向加噪、反向去噪、DDPM/DDIM 采样、score/ε 预测
      > 一句话结论：闭式解 \(\mathbf{x}_t=\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}\) 是全篇支点——
      > 它让训练能随机取 \(t\) 一步构造样本；而 ELBO 一路化简后只剩一行 MSE，
      > 预测噪声而非原图使任务难度在 \(t\) 上均衡。多步的根本原因是：\(\beta_t\) 小，反向分布才近似高斯。
- [ ] `flow-matching.md` —— Flow matching / rectified flow：与 diffusion 的关系、为何 SD3/Flux 采用
- [ ] `latent-diffusion.md` —— Latent Diffusion：VAE 潜空间、为何不在像素空间做
- [ ] `dit-arch.md` —— DiT 架构：patchify、adaLN-Zero 条件注入、scaling law（Peebles & Xie 2023）
- [ ] `conditioning.md` —— 条件机制：cross-attention、classifier-free guidance、T5/CLIP 文本编码
      > 直接承接基础篇：基础篇是无条件生成 \(p(\mathbf{x})\)，这一篇讲如何用文本控制成 \(p(\mathbf{x}\mid\mathbf{c})\)。
- [ ] `video-dit.md` —— 视频生成：时空 patch、spatial/temporal attention、时序一致性（Sora 类）
- [ ] `models-survey.md` —— 代表模型对照：SD3 / PixArt / Flux / Sora / Seedance 的取舍

## 奠基论文

- DDPM (Ho et al. 2020)、DDIM (Song et al. 2021)
- Latent Diffusion / Stable Diffusion (Rombach et al. 2022)
- **DiT**: *Scalable Diffusion Models with Transformers* (Peebles & Xie 2023)
- Flow Matching (Lipman et al. 2023) / Rectified Flow (Liu et al. 2023)
- Sora technical report (2024, 概念性)

> 完成一个主题后把 `[ ]` 改成 `[x]` 并留一句话结论。
