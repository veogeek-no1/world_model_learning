# 世界模型研究笔记

系统性地学习**多模态理解与生成**，最终指向 **世界模型 (World Model)** 这一北极星。

## 研究版图

三条主线，前两条是地基，第三条是目标：

| | 感知/理解 | 生成/想象/模拟 | 动力学/预测 |
|---|---|---|---|
| **方向** | VLM | DiT | 世界模型内核 |
| **映射** | 图像 → 文字 | 文字/条件 → 图像/视频 | 状态 + 动作 → 未来状态 |
| **回答** | "这是什么" | "构造一个世界" | "接下来会发生什么" |

**核心信念**：视频生成模型若要产出时序连贯、符合物理的内容，必须隐式学到物体永恒性、遮挡、动力学、因果。因此 VLM 的理解能力 + DiT 的生成能力，是通往世界模型的两块地基。

## 从哪里开始

如果只读一篇，读 **[演进脉络](roadmap.md)** —— 它用一条主线串起 DDPM → DDIM → LDM → DiT → Flow Matching → 视频 DiT → 世界模型，每一步都框定为"在解决上一步的哪个瓶颈"。深入任何单个节点之前，建议先在这张图上定位它。

三条主线各自的主题清单与奠基论文：

- **[VLM · 视觉理解](vlm/README.md)** —— 视觉编码器、模态对齐、LLaVA 系、评测与幻觉
- **[DiT · 生成与模拟](dit/README.md)** —— diffusion 基础、flow matching、DiT 架构、视频时空建模
- **[世界模型 · 北极星](world-model/README.md)** —— Sora as simulator、Genie、JEPA 三条路线之争

## 概念澄清

几个容易混淆的点，先钉死：

- **VLM ≠ vLLM** —— VLM 是"看图说话"的模型；vLLM 是跑模型的推理引擎（核心是 PagedAttention），属于系统层基础设施，不在本研究范围内。
- **DiT 换的是去噪骨干** —— diffusion 的加噪 → 去噪 → 采样框架不变，DiT 只是把去噪网络从 U-Net 换成 Transformer，换来可预测的 scaling law。
- **"diffusion" 常是泛称** —— SD3、Flux 等实际用的是 flow matching / rectified flow，严格说应称"Transformer backbone + flow-matching 目标"。

---

!!! note "关于这个站点"

    内容由 [仓库](https://github.com/veogeek-no1/world_model_learning) 中的 Markdown 自动构建。笔记仍在持续补充中，主题清单里未勾选的条目表示尚未动笔。
