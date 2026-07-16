# 世界模型研究笔记 (World Model Research)

这个目录用于系统性地学习和研究**多模态理解与生成**，最终指向**世界模型 (World Model)** 这一北极星。

## 研究版图

三条主线，前两条是地基，第三条是目标：

```
        感知/理解              生成/想象/模拟           动力学/预测
        (Understanding)        (Generation)            (World Model)
           VLM                    DiT                   世界模型内核
        图像 → 文字            文字/条件 → 图像/视频     状态+动作 → 未来状态
        "这是什么"             "构造一个世界"           "接下来会发生什么"
```

- **VLM（vlm/）**：Vision-Language Model，图像/视频 → 文字。视觉理解与语言接地(grounding)。
  代表：LLaVA、Qwen-VL、InternVL、GPT-4V。核心议题：视觉编码器、模态对齐、指令微调、评测。
- **DiT（dit/）**：Diffusion Transformer，文字/条件 → 图像/视频。生成与模拟。
  代表：DiT、Stable Diffusion 3、PixArt、Flux、Sora、可灵、Seedance。核心议题：diffusion/flow-matching 基础、Transformer backbone、latent space、视频时空建模。
- **世界模型（world-model/）**：把理解 + 生成 + 动力学统一起来，学习并模拟环境的演化。
  代表论点：Sora as "world simulator"、Genie（可交互）、JEPA 系。核心议题：长时序一致性、可交互模拟、物理/因果的隐式学习。

**核心信念**：视频生成模型若要产出时序连贯、符合物理的内容，必须隐式学到物体永恒性、遮挡、动力学、因果——因此 VLM 的理解能力 + DiT 的生成能力，是通往世界模型的两块地基。

## 概念澄清（避免混淆）

- **vLLM ≠ 本项目**：vLLM 是 LLM 推理引擎（工程优化，核心 PagedAttention），是系统层的基础设施，不是本研究方向。
- **VLM ≠ vLLM**：VLM 是"看图说话"的模型；vLLM 是跑模型的引擎。
- **DiT 换的是去噪骨干**：diffusion 的加噪→去噪→采样框架不变，DiT 把去噪网络从 U-Net 换成 Transformer，带来可预测的 scaling。
- **"diffusion" 常是泛称**：SD3/Flux 等实际用 flow matching / rectified flow，严格说是"Transformer backbone + flow-matching 目标"。

## 对 Claude 的工作约定

- **重算法思想与核心机制**：解释某个模型/方法时，先讲清"要解决什么问题、核心 insight 是什么、数据如何流动"，再给公式与实现细节。避免停留在 API/用法层面。
- **对照论文与架构演进**：把每个方法放进它的技术脉络里（前作解决不了什么 → 本作的关键改动 → 留下什么新问题），而不是孤立讲。
- **公式与直觉并重**：diffusion/flow-matching 这类涉及数学的，既给出核心公式，也给出物理直觉与几何图像。
- **诚实标注不确定性**：闭源模型（Sora、Seedance 等）的内部细节不可凭记忆断言；前沿方向迭代快。不确定时明说，并建议联网核对技术报告/论文。
- **笔记沉淀**：研究结论写入对应主线目录下的笔记文件，而非只停留在对话。论文精读放 `papers/`。
- **中文优先**：默认中文讲解，专有名词（diffusion、flow matching、latent、cross-attention 等）保留英文。

## 目录结构

```
world_model_research/
├── CLAUDE.md          # 本文件
├── vlm/               # 图像→文字：视觉理解
│   └── README.md      # 该主线的主题清单与进度
├── dit/               # 文字→图像/视频：生成与模拟
│   └── README.md
├── world-model/       # 北极星：理解+生成+动力学
│   └── README.md
└── papers/            # 各方向奠基论文原文/精读笔记
```

## 环境说明

- 当前目录尚未初始化为 git 仓库；开始记笔记后可 `git init` 做版本管理。
- 本目录默认以**论文精读 + 算法/架构分析**为主，暂不假设有训练/推理的 GPU 环境。
