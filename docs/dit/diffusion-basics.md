# Diffusion 基础：前向加噪、反向去噪、训练与采样

!!! abstract "这一篇要回答什么"

    - 生成模型为什么绕这么大一圈，非要"先毁掉图像再学着修复"？
    - 为什么前向过程可以**一步跳到任意时刻** \(t\)，不用真的循环加噪一千次？
    - 那个看起来朴素到不像话的损失 \(\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\|^2\)，是怎么从 ELBO 一路化简出来的？
    - 为什么网络要预测**噪声**，而不是直接预测干净图像？

    对应论文：DDPM (Ho et al., 2020)、DDIM (Song et al., 2021)、Improved DDPM (Nichol & Dhariwal, 2021)。

## 1. 出发点：生成建模难在哪

生成建模的任务是：给定一堆样本（比如"所有自然图像"），学到它们的分布 \(p(\mathbf{x})\)，并能从中**采样**出新样本。

难点在于 \(p(\mathbf{x})\) 极其复杂。一张 \(256\times256\) 的 RGB 图像住在 20 万维空间里，而"看起来像真实照片"的那些点，只占据其中一个维度极低、形状极其扭曲的流形。要用一个神经网络一步到位地把简单分布（高斯球）映射到这个流形上，是件很难的事——前作的伤疤都在这：

| 路线 | 做法 | 硬伤 |
|---|---|---|
| **GAN** | 生成器 vs 判别器对抗 | 训练不稳定、mode collapse，覆盖不全 |
| **VAE** | 编码到隐变量再一步解码 | 样本偏糊（高斯似然 + 后验近似的双重代价）|
| **自回归** | 逐像素条件生成 | 数学干净，但生成一张图要几万次前向，太慢 |

??? note "展开：GAN 和 VAE 的硬伤，根因到底是什么（这决定了 diffusion 能躲开什么）"

    一句话先行：**GAN 的病根在"它不是最小化一个损失，而是打一场博弈"；VAE 的病根在"它用逐像素 L2 去拟合一个本质多解的问题"。** 看懂这两点，才能看懂后面 diffusion 为什么两个都能躲开。

    **▍GAN：为什么不稳定、为什么 mode collapse**

    GAN 的目标是个 minimax 博弈：\(\min_G \max_D\ \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z}}[\log(1-D(G(\mathbf{z})))]\)。

    **不稳定**：普通网络训练是最小化一个固定损失，永远朝下坡走，方向明确。GAN 要找的却是生成器 \(G\) 与判别器 \(D\) 的纳什均衡，数学上是个**鞍点**——优化 \(G\) 的同时 \(D\) 也在动，\(G\) 要爬的地形被 \(D\) 每一步重新塑形，反之亦然。梯度下降-上升在鞍点附近极易**震荡甚至发散**，像石头剪刀布一样打转，不收敛。

    更深一层（Arjovsky 2017）：自然图像落在高维空间的一个低维流形上，真实分布与生成分布的流形**几乎必然不重叠**。此时一个足够强的 \(D\) 能把两者完美分开，而在它饱和处（自信输出 0 或 1）传给 \(G\) 的梯度**趋近于零**。用散度的语言说：最优 \(D\) 下 GAN 在最小化两分布的 JS 散度，而两者不重叠时 JS 恒为常数 \(\log 2\)，梯度处处为零——\(G\) 收不到"往哪挪"的有效信号。于是陷入刀刃上的两难：\(D\) 太弱指导不了 \(G\)，太强又让 \(G\) 梯度消失，要人工把两者强弱精确平衡住。这就是 GAN 难训的本质。（Wasserstein GAN 换用 Earth-Mover 距离，正是为了让不重叠时仍有平滑梯度。）

    **mode collapse**：\(G\) 的唯一任务是骗过 \(D\)。只要它找到几张能稳定骗过 \(D\) 的输出，就**没有任何动力去覆盖数据的全部多样性**——损失里根本没有一项在惩罚"你漏掉了某个模式"。对照之下，基于似然的方法（VAE、diffusion、自回归）大致在最小化 \(\mathrm{KL}(p_{\text{data}}\,\|\,p_{\text{model}})\)，这是**覆盖型（mode-covering）**的：凡真实数据有、而模型概率为零处，KL 会爆成无穷，逼模型覆盖所有模式。**似然=覆盖，对抗=寻峰**，这就是 mode collapse 的根。

    **▍VAE：为什么样本糊**

    VAE 的 decoder \(p_\theta(\mathbf{x}\mid\mathbf{z})\) 通常建成一个固定方差的高斯，于是 \(\log p_\theta(\mathbf{x}\mid\mathbf{z}) \propto -\|\mathbf{x}-\text{decoder}(\mathbf{z})\|^2\)——最大化它就是**最小化逐像素 L2**。

    关键在这：给定一个 \(\mathbf{z}\)，合理的 \(\mathbf{x}\) 往往**不止一张**（边缘、纹理的精确位置本就有歧义，是一对多映射）。而 L2 在目标多峰时，最优解是所有合理输出的**条件均值**；把若干张各自清晰、位置略错位的图平均起来，得到的就是一张糊图。逐像素高斯无法表达"边缘要么在这、要么在那"这种多峰，它只会**折中**——这是 VAE 糊的根本原因。

    次因是**后验近似的 gap**：真实后验 \(p(\mathbf{z}\mid\mathbf{x})\) 很复杂，VAE 用简单的对角高斯 \(q_\phi(\mathbf{z}\mid\mathbf{x})\) 去近似它。ELBO 只是对数似然的**下界**，差额恰是 \(\mathrm{KL}(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z}\mid\mathbf{x}))\)，因 \(q\) 被限制在简单分布族里而不为零——你优化的是一个松的下界，拟合能力天然打了折。

    **▍于是 diffusion 的立足点**

    - **躲开 GAN 的两个病**：diffusion 基于似然（覆盖型，不 collapse），且训练是**单一的 MSE 回归**，永远朝下坡走——没有博弈、没有鞍点、没有平衡 \(D\) 强弱的刀刃。这正是它稳定性上碾压 GAN 的原因。
    - **躲开 VAE 的糊**：这里有个漂亮的对称。VAE 和 diffusion **用的是同一件工具**——高斯条件 + 逐像素 MSE，区别只在**用对没用对地方**。VAE 想**一步**从 \(\mathbf{z}\) 解码到 \(\mathbf{x}\)，而这一步的条件分布高度多峰，高斯假设严重失效 → 取均值 → 糊。diffusion 把过程拆成上千小步，每步只去掉一点点噪声，此时反向条件 \(p(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\) **真的近似单峰高斯**（见第 4 节的论证），预测它的均值不会跨多张迥异的图去平均，自然不糊。**同一个高斯 MSE，VAE 用在了它失效的 regime，diffusion 用在了它成立的 regime**——这正是"为什么必须拆成很多小步"的一半意义。

??? note "再展开：为什么「最大似然 = 覆盖型」？彻底讲清前向 / 反向 KL 的不对称"

    上一块用到一句"似然=覆盖、对抗=寻峰"。这句话是理解 VAE / diffusion / GAN 分野的根，但它一点也不显然，值得单独讲透。核心只有一件事：**KL 散度是不对称的，而最大似然恰好用的是它"覆盖型"的那一侧。**

    **▍先回顾：最大似然（MLE）到底是什么**

    别急着谈"两个分布"——MLE 的定义里根本没有第二个分布，它就是**调模型参数、让你手上的观测数据最可能**。给定观测样本 \(x_1,\dots,x_N\)，我们假设一个带未知参数 \(\theta\) 的分布 \(q_\theta\)。先把这个符号钉死：**\(q_\theta\) 就是我们对"原始数据的概率分布"的参数化估计**——真实分布 \(p\) 未知也拿不到，所以造一个带旋钮的分布来当它的替身，旋钮（\(\theta\)）拧到哪，替身就长成什么样。**似然**则是把数据看成 \(\theta\) 的函数、问"这组参数下我实际看到的数据有多可能"：

    \[
    L(\theta) = \prod_i q_\theta(x_i),\qquad \theta^* = \arg\max_\theta \sum_i \log q_\theta(x_i)
    \]

    一个要记牢的区别：同一个 \(q_\theta(x)\)，**固定 \(\theta\) 看 \(x\)** 是**概率**，**固定数据 \(x\) 看 \(\theta\)** 才是**似然**（它不是 \(\theta\) 上的分布，对 \(\theta\) 积分不为 1）。MLE 用的是后者——这正是它叫"似然估计"而非"概率估计"的原因。

    最小的例子：抛硬币 10 次得 7 正，模型是正面概率为 \(p\) 的伯努利，似然 \(L(p)=p^7(1-p)^3\)，最大化得 \(p^*=0.7\)——那个"让 7/10 正面这件事最可能发生"的 \(p\)。**全程只有数据 + 一个带旋钮的模型，没有第二个分布。**

    那"逼近真实分布"是从哪冒出来的？下面第一步就会看到：它是样本量趋于无穷时**推出来的等价结果**，不是定义。把神经网络的几百万权重看成硬币那个 \(p\)、把训练图像看成那 10 次抛掷，故事完全一样。

    **▍第一步：最大似然就是在最小化"前向 KL"**

    记真实分布为 \(p\)（数据），模型为 \(q_\theta\)。最大似然要最大化 \(\mathbb{E}_{\mathbf{x}\sim p}[\log q_\theta(\mathbf{x})]\)。而

    \[
    \mathrm{KL}(p\,\|\,q_\theta) = \mathbb{E}_{p}[\log p(\mathbf{x})] - \mathbb{E}_{p}[\log q_\theta(\mathbf{x})] = -H(p) - \mathbb{E}_{p}[\log q_\theta(\mathbf{x})]
    \]

    第一项 \(-H(p)\) 是真实分布的熵，与 \(\theta\) 无关。所以

    \[
    \max_\theta\ \mathbb{E}_{p}[\log q_\theta] \iff \min_\theta\ \mathrm{KL}(p\,\|\,q_\theta)
    \]

    **最大似然 = 最小化前向 KL**（"前向"指真实分布 \(p\) 摆在第一个位置、期望对 \(p\) 取）。这一步是后面一切的基础。

    **▍第二步：前向 KL 的不对称性 —— 为什么它"零回避 / 覆盖"**

    盯住被积函数看：\(\mathrm{KL}(p\|q) = \int p(\mathbf{x})\,\log\frac{p(\mathbf{x})}{q(\mathbf{x})}\,d\mathbf{x}\)。分两种区域：

    - **数据有、模型无**（\(p(\mathbf{x})>0\) 但 \(q(\mathbf{x})\to 0\)）：\(\log\frac{p}{q}\to+\infty\)，又被正的 \(p(\mathbf{x})\) 加权 → 积分**炸到 \(+\infty\)**。**重罚。**
    - **数据无、模型有**（\(p(\mathbf{x})=0\) 但 \(q(\mathbf{x})>0\)）：被积函数 \(=0\cdot\log(0/q)=0\) → **完全不罚**。

    一句话：前向 KL **只怕"漏"，不怕"多"**。模型必须在所有数据出现过的地方都放上概率质量（否则损失爆炸），却可以放心地把质量溢出到没有数据的空白区（免费）。这就是**零回避（zero-avoiding）= 覆盖型（mode-covering）**：宁可铺得太宽、把空谷也占上，也绝不敢丢掉任何一个数据模式。

    **▍第三步：反向 KL —— 反过来，"零强制 / 寻峰"**

    换成 \(\mathrm{KL}(q\|p) = \int q(\mathbf{x})\log\frac{q(\mathbf{x})}{p(\mathbf{x})}\,d\mathbf{x}\)（期望改对 \(q\) 取），不对称性整个翻过来：

    - **模型有、数据无**（\(q>0,\ p\to0\)）：\(\log\frac{q}{p}\to+\infty\)，被 \(q\) 加权 → 炸。**重罚** → 模型不敢在没有数据处放质量。
    - **数据有、模型无**（\(p>0,\ q=0\)）：被积函数 \(=0\) → **不罚漏掉的模式**。

    所以反向 KL **只怕"多"，不怕"漏"**：宁可缩进一个峰里、又窄又尖，也绝不把质量撒到峰与峰之间的低密度区。这是**零强制（zero-forcing）= 寻峰型（mode-seeking）**。

    **▍一张图看穿：单个高斯拟合双峰目标**

    ![单个高斯拟合双峰目标：前向 KL 覆盖 vs 反向 KL 寻峰](images/kl-asymmetry-light.svg#only-light)
    ![单个高斯拟合双峰目标：前向 KL 覆盖 vs 反向 KL 寻峰](images/kl-asymmetry-dark.svg#only-dark)

    同一个双峰目标 \(p\)（灰），用同一族（单个高斯）去拟合，两种 KL 给出截然相反的解：

    - **前向 KL（蓝，最大似然）**：最优解是**矩匹配**的宽高斯（均值 0、标准差 2.09），骑跨两峰，连中间 \(x=0\) 的空谷都放了质量——覆盖到了，但溢进了没有数据的地方。
    - **反向 KL（橙）**：锁定其中一个峰（均值 −2、标准差 0.6），又窄又尖，**另一个峰被彻底丢弃**——寻峰。

    （顺带一提：前向 KL 对指数族的最优解正是**矩匹配**，这是"覆盖型"最干净的数学写照。）

    **▍回到生成模型**

    - **VAE、diffusion、自回归** = （变分）最大似然 = 前向 KL = **覆盖型**：努力覆盖数据的全部模式，代价是可能在低密度区也放一点质量。
    - **GAN** = 对抗目标，实践中的非饱和损失与训练动态**偏向寻峰**（严格说，最优判别器下 GAN 最小化的是 JS 散度、并非干净的反向 KL，这里只取其"寻峰"的行为倾向，不做等号）。所以 GAN 样本常常**又清晰又不全**——这正是 mode collapse。

    **▍关键澄清：覆盖型 ≠ 糊**

    很容易误读成"覆盖型 → 糊"，但这是两件事，务必分开：

    - **覆盖**说的是**哪些峰被表示**（全都表示 vs 只挑一个）——**分布层面**的性质。
    - **糊**说的是**每个条件怎么被画出来**——**渲染层面**的性质，来自"用单峰高斯去拟合一个本身多峰的条件、于是被迫取均值"。

    **diffusion 和 VAE 都是覆盖型，但 diffusion 不糊、VAE 糊**，差别只在渲染层面：VAE 想一步从 \(\mathbf{z}\) 解码，那一步的条件 \(p(\mathbf{x}\mid\mathbf{z})\) 高度多峰，高斯一取均值就糊；diffusion 拆成上千小步，每步反向条件近似单峰（第 4 节），高斯取均值不跨越迥异图像，于是**既覆盖又清晰**。一句话收束：**覆盖型是好事（不丢模式），糊是另一个问题（单峰高斯用错了地方），diffusion 恰好只要前者、避开后者。**

    **▍最后一块拼图：\(q_\theta(\mathbf{x})\) 能不能算出来，决定了各家怎么训**

    上面默认 MLE 可以照定义执行，但那有个前提：你得能对任意一个样本 \(\mathbf{x}\) **求出 \(q_\theta(\mathbf{x})\) 的数值**。能不能求，把生成模型分成了三档：

    - **显式可算**（伯努利、高斯、自回归）：\(q_\theta(\mathbf{x})\) 有显式公式，直接照定义 \(\max_\theta\sum_i\log q_\theta(x_i)\) 训练。自回归把 \(q_\theta(\mathbf{x})\) 拆成 \(\prod_i q_\theta(x_i\mid x_{<i})\)、每个因子都可算——第 1 节表格里说它"数学干净"，指的就是这个：**精确似然**。
    - **算不动，但够得着下界**（VAE、diffusion）：\(q_\theta(\mathbf{x})\) 要对全部隐变量积分，intractable。于是退而优化它的**变分下界（ELBO）**——第 5 节那整套推导不是故弄玄虚，正是因为真实的 \(\log q_\theta(\mathbf{x})\) 求不出来，只能优化一个够得着的下界。
    - **完全没有密度**（GAN）：只能从模型**采样**（噪声进、图像出），却问不出"这张图的密度是多少"——所谓**隐式模型（implicit model）**。没有似然可最大化，只能另起炉灶走对抗训练；也因此脱离了"似然=覆盖"的保护伞，mode collapse 由此埋下。

    一句话：**MLE 是理想，能不能照做取决于 \(q_\theta\) 的密度可不可求——可求就直接 MLE（自回归），求不动就优化下界（VAE / diffusion），根本没有密度就对抗（GAN）。**

## 2. 核心 insight：把一个难问题拆成一千个简单问题

Diffusion 的想法可以用一句话概括：

!!! quote "核心 insight"

    与其学"一步从噪声跳到图像"（太难），不如学"很多步、每步只去掉一点点噪声"（每步都简单到几乎是线性的）。

这个拆解之所以成立，靠的是一个关键的不对称性：

- **把图像毁掉是容易的**——往上加高斯噪声就行，完全不需要学习，甚至有解析解。
- **把图像修复是困难的**——但如果每一步只毁掉了"一点点"，那么修复这一点点，就是个简单到可以被神经网络轻松拟合的任务。

于是我们造一条链：前向一路加噪把图像碾成纯噪声（**固定、无参数**），反向一路去噪把纯噪声还原成图像（**这才是要学的**）。

```mermaid
flowchart TB
    subgraph F["前向过程 q&nbsp;&nbsp;（固定，无可学参数，加噪）"]
        direction LR
        A0["<b>x₀</b><br>真实图像"] -->|"加噪"| A1["x₁"] -->|"加噪"| AD["···"] -->|"加噪"| AT["<b>x_T</b><br>≈ N(0, I)"]
    end
    subgraph R["反向过程 p_θ&nbsp;&nbsp;（神经网络要学的，去噪）"]
        direction LR
        BT["<b>x_T</b><br>随机噪声"] -->|"去噪"| BD["···"] -->|"去噪"| B1["x₁"] -->|"去噪"| B0["<b>x₀</b><br>生成图像"]
    end
    F -.->|"训练时：用前向廉价地造出 (x_t, ε) 监督对，教网络认出 ε"| R

    style A0 fill:#4c6ef5,stroke:#364fc7,color:#fff
    style AT fill:#e8710a,stroke:#c25708,color:#fff
    style BT fill:#e8710a,stroke:#c25708,color:#fff
    style B0 fill:#4c6ef5,stroke:#364fc7,color:#fff
```

采样时我们只用反向链；前向链的唯一作用，是在训练期**廉价地伪造出无穷多的训练数据**——任取一张真图、任取一个 \(t\)、任取一个噪声，就得到一组 \((\mathbf{x}_t, \boldsymbol{\epsilon})\) 监督对。

## 3. 前向过程：加噪，以及那个关键的闭式解

### 3.1 定义

每一步往图像里掺一点高斯噪声，同时把原信号按比例缩小一点：

\[
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I}\right)
\]

其中 \(\{\beta_t\}_{t=1}^T\) 是预先设定的 **noise schedule**，\(\beta_t \in (0,1)\) 且通常随 \(t\) 递增。DDPM 原文取 \(T=1000\)，\(\beta_t\) 从 \(10^{-4}\) 线性升到 \(0.02\)。

写成采样形式更直观（重参数化）：

\[
\mathbf{x}_t = \sqrt{1-\beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
\]

!!! question "为什么缩放系数偏偏是 \(\sqrt{1-\beta_t}\)？"

    为了**保持方差**。假设 \(\mathrm{Var}(\mathbf{x}_{t-1}) = 1\)（数据已归一化），那么

    \[
    \mathrm{Var}(\mathbf{x}_t) = (1-\beta_t)\cdot 1 + \beta_t = 1
    \]

    信号被压缩多少，就正好补进多少噪声，总能量守恒。这类 schedule 因此被称为 **variance preserving (VP)**。若不做这个缩放，随着 \(t\) 增大方差会一路膨胀到爆炸，网络需要处理的数值尺度在不同 \(t\) 之间差几个数量级，根本没法训练。

### 3.2 闭式解：一步跳到任意时刻 t

如果每次采样 \(\mathbf{x}_t\) 都要老老实实循环 \(t\) 次，训练会慢到不可用。幸好高斯的叠加有解析解。记

\[
\alpha_t := 1-\beta_t, \qquad \bar\alpha_t := \prod_{s=1}^{t}\alpha_s
\]

展开两步看看：

\[
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}_{t-1}\\
&= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}\,\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}}\,\boldsymbol{\epsilon}_{t-2}\right) + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}_{t-1}\\
&= \sqrt{\alpha_t\alpha_{t-1}}\,\mathbf{x}_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\boldsymbol{\epsilon}_{t-2} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}_{t-1}}_{\text{两个独立高斯之和}}
\end{aligned}
\]

关键的一步：两个独立零均值高斯相加，结果仍是高斯，方差直接相加：

\[
\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\alpha_{t-1}
\]

所以那一大坨等价于单个 \(\sqrt{1-\alpha_t\alpha_{t-1}}\,\bar{\boldsymbol{\epsilon}}\)。归纳下去即得**本篇最重要的公式**：

\[
\boxed{\ \mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})\ }
\]

即 \(q(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\sqrt{\bar\alpha_t}\mathbf{x}_0,\ (1-\bar\alpha_t)\mathbf{I})\)。

这个式子的意义怎么强调都不过分：

- **训练可以随机取 \(t\)**，一次前向就构造出样本，无需循环 —— 这是 diffusion 能训得动的前提。
- \(\sqrt{\bar\alpha_t}\) 就是**信号残留的比例**，\(\sqrt{1-\bar\alpha_t}\) 是**噪声占比**，两者平方和恒为 1。整个前向过程被压缩成了一个"信号与噪声此消彼长"的插值。
- \(\bar\alpha_t\) 单调递减；当 \(\bar\alpha_T \approx 0\) 时 \(\mathbf{x}_T\approx\mathcal{N}(\mathbf{0},\mathbf{I})\)，与 \(\mathbf{x}_0\) 无关 —— 这正是采样时能从纯噪声起步的依据。

### 3.3 Noise schedule：信号该以什么节奏衰减

既然一切由 \(\bar\alpha_t\) 决定，它的形状就是个设计选择。DDPM 的线性 schedule 有个后来被发现的毛病：

![linear 与 cosine schedule 的对比](images/noise-schedule-light.svg#only-light)
![linear 与 cosine schedule 的对比](images/noise-schedule-dark.svg#only-dark)

/// caption
左：信号残留比例 \(\bar\alpha_t\) 的衰减。右：对数信噪比 \(\log\mathrm{SNR}_t = \log\frac{\bar\alpha_t}{1-\bar\alpha_t}\)。
///

线性 schedule（橙）在 \(t\approx 259\) 就已经丢掉一半信号，到后半程 \(\bar\alpha_t\) 早已贴地——最后 100 步的 \(\bar\alpha_t\) 平均只有 \(1.2\times10^{-4}\)，几乎是在纯噪声上空转。这些步数既没在破坏信息（早毁完了），也就没给网络提供有效的学习信号，等于白白浪费了近四分之一的采样预算。

Improved DDPM 因此改用 **cosine schedule**（蓝）：

\[
\bar\alpha_t = \frac{f(t)}{f(0)},\qquad f(t)=\cos^2\!\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right),\quad s=0.008
\]

它到 \(t\approx 496\)（差不多正中间）才丢掉一半信号，末段 \(\bar\alpha_t\) 均值 \(7.9\times10^{-3}\)，比线性高约 60 倍，信息销毁得更均匀，每一步都在干活。实现上还要把由此反推出的 \(\beta_t\) 截断在 0.999 以内，否则末步 \(\bar\alpha_T\) 会精确塌到 0 导致方差退化。

!!! tip "复现这张图"

    图由 [`scripts/gen_noise_schedule.py`](https://github.com/veogeek-no1/world_model_learning/blob/main/scripts/gen_noise_schedule.py) 生成，
    上面这些数字全部来自实际计算而非估计。改 schedule 参数后重跑 `python scripts/gen_noise_schedule.py` 即可更新。

## 4. 反向过程：为什么它也是高斯

我们想要 \(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\)——但它需要对整个数据分布积分，无法求得。这里有个漂亮的理论结果救场：

!!! note "关键事实"

    当 \(\beta_t\) 足够小时，反向条件分布 \(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\) **也近似是高斯**。

    这正是"必须拆成很多小步"的深层原因：步子迈大了，反向分布会变成复杂的多峰分布（从一张糊图能还原出的清晰图有很多种），高斯就拟合不了了。**\(T\) 大 \(\Leftrightarrow\) 每步 \(\beta_t\) 小 \(\Leftrightarrow\) 反向可用高斯近似**——三者是一回事。

??? note "展开：这个「关键事实」凭什么成立？三层论证，从机制到定理"

    上面是个断言，不该被无条件吞下。它有三层论证，严格程度递增：Taylor 展开讲机制，一个可精确求解的例子给硬验证，连续时间定理给严格地基。

    **▍论证一：Bayes + Taylor —— 看清"小步"到底在哪里进场**

    对反向条件用贝叶斯定理，先把公式写全：

    \[
    q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\ =\ \frac{q(\mathbf{x}_t\mid\mathbf{x}_{t-1})\ \cdot\ q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}
    \]

    三个因子各是谁：分子第一项是**似然**——"上一步若是这张图，加一步噪声后变成现在这样的概率"，这是前向方向，我们自己定义的，明确是高斯；分子第二项是**边缘** \(q(\mathbf{x}_{t-1})\)——\(t-1\) 噪声水平上这张图本身的常见程度，复杂多峰，扮演先验的角色；分母 \(q(\mathbf{x}_t)\) 是**归一化项**。

    关键一步：**\(\mathbf{x}_t\) 是已经观测到、固定住的**，所以分母 \(q(\mathbf{x}_t)\) 不随 \(\mathbf{x}_{t-1}\) 变，只是一个常数。我们关心的是后验作为 \(\mathbf{x}_{t-1}\) 的函数长什么**形状**，常数不影响形状（最后归一化补回即可），因此把它省掉，写成正比：

    \[
    q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\ \propto\ \underbrace{q(\mathbf{x}_t\mid\mathbf{x}_{t-1})}_{\text{似然：高斯，已知}}\ \cdot\ \underbrace{q(\mathbf{x}_{t-1})}_{\text{边缘：复杂多峰}}
    \]

    **似然项**作为 \(\mathbf{x}_{t-1}\) 的函数：

    \[
    \log q(\mathbf{x}_t\mid\mathbf{x}_{t-1}) = -\frac{1-\beta_t}{2\beta_t}\left\|\mathbf{x}_{t-1}-\frac{\mathbf{x}_t}{\sqrt{1-\beta_t}}\right\|^2 + C
    \]

    是一个中心在 \(\mathbf{x}_t\) 附近、**宽度 \(O(\sqrt{\beta_t})\) 的极窄高斯窗**——它把 \(\mathbf{x}_{t-1}\) 死死限制在 \(\mathbf{x}_t\) 周围的小球里。

    **边缘项** \(\log q(\mathbf{x}_{t-1})\) 是复杂多峰的函数，但我们**只需要它在窗内的样子**。在窗内 Taylor 展开：

    \[
    \log q(\mathbf{x}_{t-1}) \approx \log q(\mathbf{x}_t) + (\mathbf{x}_{t-1}-\mathbf{x}_t)^\top\nabla\log q(\mathbf{x}_t) + \tfrac{1}{2}(\mathbf{x}_{t-1}-\mathbf{x}_t)^\top\mathbf{H}\,(\mathbf{x}_{t-1}-\mathbf{x}_t)
    \]

    逐项看它对高斯窗做了什么：

    - **线性项**：高斯乘一个指数线性倾斜**仍是高斯**，只是均值平移 \(\beta_t\nabla\log q(\mathbf{x}_t)\)。注意——**score 自己冒出来了**，反向均值本质上就是"沿 score 场走一小步"，这正好呼应第 7 节。
    - **二次项**：边缘分布的曲率 \(\mathbf{H}\) 是 \(O(1)\)（由数据分布的平滑度决定，不随 \(\beta_t\) 变），而似然窗的曲率是 \(1/\beta_t\)。相对修正 \(=O(\beta_t)\to 0\)，忽略。

    于是（\(q_{t-1}\) 与 \(q_t\) 之差同为 \(O(\beta_t)\)，并入误差项）：

    \[
    q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\ \approx\ \mathcal{N}\!\left(\frac{\mathbf{x}_t}{\sqrt{1-\beta_t}}+\beta_t\nabla\log q(\mathbf{x}_t),\ \ \beta_t\mathbf{I}\right)
    \]

    **"小步"在哪进场，一目了然**：整个论证靠"在 \(O(\sqrt{\beta_t})\) 宽的窗内 \(\log q\) 可以线性化"。步子一大，窗变宽，窗内装进边缘分布的多个峰，线性化失效 → 后验真的变多峰。"一张糊图能还原出很多种清晰图"，说的就是窗太宽、窗里有好几个互不相干的原像。

    **▍论证二：一个可精确求解的例子 —— 双点数据，亲手验证**

    取最极端的数据分布：只有两个点，\(p_{\text{data}}=\tfrac12\delta_{-1}+\tfrac12\delta_{+1}\)。此时反向条件可以**精确算出**（就用本节下文的后验公式 \(\tilde\mu_t,\tilde\beta_t\)，对两个可能的 \(\mathbf{x}_0\) 加权）：

    \[
    q(x_{t-1}\mid x_t)=\sum_{k=\pm1} w_k(x_t)\,\mathcal{N}\!\big(\tilde\mu_t(x_t,k),\ \tilde\beta_t\big),\qquad w_k\propto\tfrac12\,\mathcal{N}\!\big(x_t;\ \sqrt{\bar\alpha_t}\,k,\ 1-\bar\alpha_t\big)
    \]

    它**天生是两个高斯的混合**——严格说就是多峰的！但看"分量间距 / 分量宽度"这个比值：\(\tilde\mu_t\) 里 \(x_0\) 的系数是 \(\sqrt{\bar\alpha_{t-1}}\beta_t/(1-\bar\alpha_t)\)，所以**间距 \(=O(\beta_t)\)**；而分量宽度 \(\sqrt{\tilde\beta_t}=O(\sqrt{\beta_t})\)。比值 \(=O(\sqrt{\beta_t})\to 0\)：**间距塌缩得比宽度快，两峰融成一个高斯**。这就是"近似高斯"的精确含义。实算一遍（固定噪声水平 \(\bar\alpha_t=0.5\)、固定观测 \(x_t\)，只改单步大小）：

    ![双点数据下精确的反向条件分布：小步近似高斯，大步明显双峰](images/reverse-gaussian-light.svg#only-light)
    ![双点数据下精确的反向条件分布：小步近似高斯，大步明显双峰](images/reverse-gaussian-dark.svg#only-dark)

    - \(\beta_t=0.02\)：间距/宽度 \(=0.41\)，精确分布与单高斯拟合的 \(\mathrm{KL}=1.0\times10^{-6}\)——**肉眼与数值上都无法区分**。
    - \(\beta_t=0.4\)：间距/宽度 \(=4.0\)，\(\mathrm{KL}=0.17\)——差了**约 17 万倍**，明显双峰，单高斯拟合失效。

    反过来推到极限也成立：**一步到位**（\(\bar\alpha_{t-1}=1\)）时公式给出分量宽度 \(\to 0\)、均值正好落在 \(\pm1\)——后验变成**两根尖刺**，最极端的多峰。"步大 → 多峰"从直觉变成了精确结论。（图与数字由 [`scripts/gen_reverse_gaussian.py`](https://github.com/veogeek-no1/world_model_learning/blob/main/scripts/gen_reverse_gaussian.py) 实算生成。）

    **▍论证三：连续时间的严格定理（Anderson 1982）**

    把前向过程看成 SDE \(\mathrm{d}\mathbf{x}=f\,\mathrm{d}t+g\,\mathrm{d}\mathbf{w}\) 的离散化，则有经典的**时间反演定理**：扩散 SDE 的时间反演**仍是扩散 SDE**——

    \[
    \mathrm{d}\mathbf{x}=\big[f(\mathbf{x},t)-g^2(t)\,\nabla_\mathbf{x}\log q_t(\mathbf{x})\big]\mathrm{d}t+g(t)\,\mathrm{d}\bar{\mathbf{w}}
    \]

    而扩散 SDE 的无穷小转移**由构造就是高斯**（漂移 \(\cdot\,\mathrm{d}t\) + 高斯噪声 \(\cdot\sqrt{\mathrm{d}t}\)）。所以"小步下反向条件近似高斯"的严格版本就是：**反向过程本身也是一个扩散过程**。DDPM 的祖先采样正是这条反向 SDE 的 Euler–Maruyama 离散化——论证一推出的均值与定理的漂移逐项对得上。这条识别由 Score SDE (Song et al. 2021) 挑明；更早的出处链是 Sohl-Dickstein 2015 引 Feller (1949) 对离散情形的论证。诚实标注：Anderson 定理本身的证明需要平滑性/可积性等技术条件，超出本笔记范围。

    三层其实是一件事的三个面：**Taylor 讲机制、双点例子给硬验证、Anderson 给严格地基**——且三者都把 score \(\nabla\log q\) 顶到台面上，与第 7 节严丝合缝。

于是用神经网络参数化一个高斯：

\[
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\ \boldsymbol{\mu}_\theta(\mathbf{x}_t,t),\ \sigma_t^2\mathbf{I}\right)
\]

虽然 \(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\) 求不出，但**多给一个条件 \(\mathbf{x}_0\)，后验就有解析解**（贝叶斯 + 配方即可推出）：

\[
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\ \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0),\ \tilde\beta_t\mathbf{I}\right)
\]

\[
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t,
\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
\]

训练时 \(\mathbf{x}_0\) 是已知的（就是那张真图），所以这个后验可以当作**监督目标**。这就是整个训练目标的支点。

## 5. 训练目标：从 ELBO 到一行 MSE

### 5.1 变分上界

和 VAE 同款套路，对负对数似然取变分上界：

\[
\mathbb{E}\left[-\log p_\theta(\mathbf{x}_0)\right] \le \mathbb{E}_q\left[-\log\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\right] =: L
\]

经过整理（关键是把联合分布按马尔可夫链拆开，并把 \(q(\mathbf{x}_{t-1}|\mathbf{x}_t)\) 用带 \(\mathbf{x}_0\) 的后验替换），\(L\) 可以分解成逐项的 KL：

\[
L = \underbrace{D_{\mathrm{KL}}\!\left(q(\mathbf{x}_T|\mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\right)}_{L_T:\ \text{无可学参数，常数}}
+ \sum_{t=2}^{T}\underbrace{D_{\mathrm{KL}}\!\left(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)\right)}_{L_{t-1}:\ \text{主项}}
\underbrace{-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0:\ \text{重建项}}
\]

\(L_T\) 不含 \(\theta\)，直接扔掉。主项 \(L_{t-1}\) 是**两个高斯之间的 KL**，有闭式解——在方差固定为 \(\sigma_t^2\) 时，它退化成两个均值的平方距离：

\[
L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\left\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t,t)\right\|^2\right] + C
\]

到这里，"学一个分布"已经变成了"回归一个均值"。

### 5.2 换元：为什么预测噪声

现在做一步关键换元。由闭式解反解出 \(\mathbf{x}_0\)：

\[
\mathbf{x}_0 = \frac{1}{\sqrt{\bar\alpha_t}}\left(\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}\right)
\]

代入 \(\tilde{\boldsymbol{\mu}}_t\) 并化简，那堆系数会奇迹般地塌缩成：

\[
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol{\epsilon}\right)
\]

这个形式在说一件很直白的事：**\(\mathbf{x}_t\) 是已知的，均值里唯一未知的东西就是 \(\boldsymbol{\epsilon}\)**。那网络干脆就去预测 \(\boldsymbol{\epsilon}\) 好了。于是让网络输出 \(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\)，并把均值参数化成同样的形状：

\[
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\right)
\]

两式相减，\(\mathbf{x}_t\) 项完全抵消，只剩噪声的差：

\[
L_{t-1} = \mathbb{E}\left[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t(1-\bar\alpha_t)}\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\right\|^2\right]
\]

DDPM 最后一刀：**把前面那个复杂的权重直接扔掉，设为 1**。得到大道至简的

\[
\boxed{\ L_{\text{simple}} = \mathbb{E}_{t\sim\mathcal{U}[1,T],\ \mathbf{x}_0,\ \boldsymbol{\epsilon}}\left[\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\!\left(\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon},\ t\right)\right\|^2\right]\ }
\]

一个 ELBO 推导，最后落地成一行 MSE。

!!! question "为什么预测 \(\boldsymbol{\epsilon}\) 比预测 \(\mathbf{x}_0\) 好？"

    数学上两者可以互相换算，**等价**——预测出 \(\boldsymbol{\epsilon}\) 就等于预测出 \(\mathbf{x}_0\)。差别在优化性质上：

    - **任务难度在 \(t\) 上是均衡的**。预测 \(\mathbf{x}_0\) 时，\(t\) 小几乎白送（图基本是干净的），\(t\) 大则近乎无解（要从纯噪声里凭空变出图）——难度跨越好几个数量级，损失尺度极不均衡。而预测 \(\boldsymbol{\epsilon}\) 时，无论 \(t\) 多大，目标始终是个标准正态样本，尺度恒定。
    - **输出分布恒定**。网络的输出目标永远是 \(\mathcal{N}(\mathbf{0},\mathbf{I})\)，不随 \(t\) 漂移，对归一化和训练稳定性都友好。
    - **扔掉权重反而更好**。那个被丢弃的权重 \(\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\) 在 \(t\) 小时很大。丢掉它相当于**降低了小 \(t\)（简单去噪）的权重、抬高了大 \(t\)（粗粒度结构）的权重**，把模型容量引导到更影响观感的全局结构上。DDPM 报告这样 FID 更好——一个"理论上不严格、实践上更优"的经典案例。

### 5.3 训练循环

化简之后，训练朴素得惊人：

```python
# 每个 step：
x0 = sample_batch()                                  # 真实图像
t  = randint(1, T, size=batch)                       # 随机时间步
eps = randn_like(x0)                                 # 目标噪声

xt = sqrt(abar[t]) * x0 + sqrt(1 - abar[t]) * eps    # 闭式解，一步到位
loss = mse(eps_theta(xt, t), eps)                    # 就这一行
loss.backward()
```

没有对抗、没有额外判别器、没有采样循环。**训练稳定性正是 diffusion 打败 GAN 的关键**。

## 6. 采样：把噪声一步步搬回图像

!!! note "留意：本篇到此都是「无条件」生成，还没有方向盘"

    看下面采样公式里的网络——它始终是 \(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\)，只吃"带噪图像"和"时间步"，**没有任何文本/条件输入**。因此从纯噪声出发得到的是 \(p(\mathbf{x})\) 的一个随机样本："一张合理的自然图像"，但画什么主体、什么构图完全不受控，给不了"画一只猫"这样的指令。

    如何把网络变成 \(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})\)、从而按文本采样 \(p(\mathbf{x}\mid\mathbf{c})\)，是**条件机制**的主题，留到 `conditioning.md` 专门讲——核心是两件事：用 **cross-attention** 把文本逐词注入去噪网络，再用 **classifier-free guidance** 把条件的影响放大到够强。历史上也正是这个顺序：DDPM (2020) 做的是无条件生成，文本控制要到 2022 年的 Latent Diffusion / Stable Diffusion 才成熟。

### 6.1 DDPM 祖先采样

训练完，从 \(\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})\) 出发，逐步去噪：

\[
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\right) + \sigma_t\mathbf{z},\qquad \mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
\]

最后一步（\(t=1\)）不加噪声。\(\sigma_t^2\) 取 \(\beta_t\) 或 \(\tilde\beta_t\) 实践中差别不大。

注意末尾那个 \(\sigma_t\mathbf{z}\)：**每步都要重新注入随机噪声**。初看很反直觉——好不容易去掉噪声，为什么又加回去？因为 \(\boldsymbol{\mu}_\theta\) 只是后验的**均值**，直接沿均值走等于每步都取众数，会塌向过度平滑的"平均脸"。加回的噪声让采样真正从分布里抽样，是多样性的来源。

**代价**：\(T=1000\) 意味着生成一张图要 1000 次网络前向。这是 diffusion 最痛的地方。

### 6.2 DDIM：把随机链改成确定性映射

DDIM 的洞察是：\(L_{\text{simple}}\) 只依赖边缘分布 \(q(\mathbf{x}_t|\mathbf{x}_0)\)，**并不要求前向过程必须是马尔可夫链**。于是可以构造一族非马尔可夫过程，它们的边缘分布完全相同（因此**训练好的模型可以直接复用，无需重训**），但采样时可以跳步：

\[
\mathbf{x}_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\left(\frac{\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)}{\sqrt{\bar\alpha_t}}\right)}_{\hat{\mathbf{x}}_0:\ \text{当前对原图的估计}} + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)}_{\text{指向}\ \mathbf{x}_{t-1}\ \text{的方向}} + \sigma_t\mathbf{z}
\]

结构非常好读：**先跳到对干净图像的估计 \(\hat{\mathbf{x}}_0\)，再按新的噪声水平退回去一点**。

令 \(\sigma_t=0\) 则随机项消失，采样变成**完全确定性**的：给定 \(\mathbf{x}_T\) 就唯一确定 \(\mathbf{x}_0\)。这带来两个后果：

- **可跳步**。既然是确定性 ODE 式的轨迹，就能用大步长求解，\(1000\to 50\) 步质量几乎不掉。
- **latent 有了语义**。\(\mathbf{x}_T\) 成为图像的确定性编码，在两个 \(\mathbf{x}_T\) 之间插值可以得到语义连续的过渡——DDPM 做不到这点。

## 7. 另一个视角：score matching

值得知道的一个联系。对闭式解求对数梯度：

\[
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0) = -\frac{\mathbf{x}_t-\sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar\alpha_t}}
\]

所以噪声预测网络和 **score function**（对数概率密度的梯度）只差一个常数：

\[
\mathbf{s}_\theta(\mathbf{x}_t,t) \approx -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)}{\sqrt{1-\bar\alpha_t}}
\]

这意味着 DDPM 训练的东西，本质上就是在做 **denoising score matching**；采样则是在 score 场里做 Langevin 式的爬升——**朝着"更像真实数据"的方向走**。Song & Ermon 的 score-based 路线与 DDPM 是同一枚硬币的两面，后来被 Score SDE 统一进了同一个连续时间框架。这个视角是理解后续 flow matching 的桥梁。

## 8. 小结与遗留瓶颈

- 前向加噪**固定无参**，闭式解 \(\mathbf{x}_t=\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}\) 让训练可以随机取 \(t\)、一步构造样本。
- 反向去噪靠神经网络；因为每步 \(\beta_t\) 很小，反向分布可用高斯近似——这是"必须多步"的根本原因。
- ELBO 一路化简，落地成一行 MSE \(\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\|^2\)；预测噪声让任务难度在 \(t\) 上均衡，扔掉权重反而提升观感质量。
- DDIM 用非马尔可夫构造把采样变确定性，\(1000\to50\) 步，且模型无需重训。

留给后续的坑，正好是接下来几篇的动机：

| 瓶颈 | 谁来解决 |
|---|---|
| 从纯噪声只能生成**任意**图像，给不了"画什么"的指令 | `conditioning.md`：cross-attention 注入文本 + classifier-free guidance |
| 采样仍需几十步，且在**像素空间**做，高分辨率算力爆炸 | `latent-diffusion.md`：搬进 VAE 潜空间 |
| 加噪时间表、离散步数都是人为设定，路径弯弯绕绕 | `flow-matching.md`：直接学速度场，走直线 |
| 去噪骨干 U-Net 是卷积时代产物，scaling 行为不明 | `dit-arch.md`：换成 Transformer |

（上面四篇尚未动笔，写完后这里会改成站内链接。）

## 参考文献

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Nichol, A., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- Song, Y., et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- Luo, C. (2022). *Understanding Diffusion Models: A Unified Perspective*. [arXiv:2208.11970](https://arxiv.org/abs/2208.11970) —— 推导细节最全的一篇综述
