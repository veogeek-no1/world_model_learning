"""生成"KL 不对称性"插图：用单个高斯去拟合一个双峰目标分布。

- 前向 KL(p‖q)（最大似然的目标）：mode-covering，最优单高斯是矩匹配的宽高斯，
  铺满两个峰，连中间空谷也占——覆盖型。
- 反向 KL(q‖p)：mode-seeking，最优单高斯锁定其中一个峰、又窄又尖——寻峰型。

输出 docs/dit/images/kl-asymmetry-{light,dark}.svg，明暗各一版，文字转矢量路径。
重新生成：python scripts/gen_kl_asymmetry.py
"""
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for p in ("/System/Library/Fonts/PingFang.ttc",
          "/System/Library/Fonts/Hiragino Sans GB.ttc",
          "/System/Library/Fonts/STHeiti Light.ttc"):
    if os.path.exists(p):
        try:
            font_manager.fontManager.addfont(p)
            matplotlib.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=p).get_name()
            ] + matplotlib.rcParams["font.sans-serif"]
            break
        except Exception:
            continue
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["svg.fonttype"] = "path"


def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


# --- 目标：双峰混合高斯 ---
MU, SIG = 2.0, 0.6
x = np.linspace(-8, 8, 4000)
dx = x[1] - x[0]
p = 0.5 * gauss(x, -MU, SIG) + 0.5 * gauss(x, MU, SIG)

# --- 前向 KL 最优单高斯 = 矩匹配（解析）---
mean_fwd = 0.0                                  # 对称，均值为 0
var_fwd = SIG ** 2 + MU ** 2                    # E[x^2] - 0 = σ² + μ²
std_fwd = np.sqrt(var_fwd)
q_fwd = gauss(x, mean_fwd, std_fwd)

# --- 反向 KL 最优单高斯：数值极小化 KL(q‖p)，网格搜索 ---
eps = 1e-12
best = None
for mu in np.linspace(-4, 4, 401):
    for sig in np.linspace(0.2, 2.5, 231):
        q = gauss(x, mu, sig)
        kl = np.sum(q * np.log((q + eps) / (p + eps))) * dx   # ∫ q log(q/p)
        if best is None or kl < best[0]:
            best = (kl, mu, sig)
_, mu_rev, std_rev = best
q_rev = gauss(x, mu_rev, std_rev)

C_TGT = "#8b949e"
C_FWD = "#4c6ef5"   # 蓝：前向 KL / 覆盖
C_REV = "#e8710a"   # 橙：反向 KL / 寻峰


def render(dark: bool, out: str) -> None:
    fg = "#c9d1d9" if dark else "#24292f"
    grid = "#8b949e" if dark else "#d0d7de"
    tgt_fill = "#3d4450" if dark else "#e7ebf0"

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.fill_between(x, p, color=tgt_fill, zorder=1)
    ax.plot(x, p, color=C_TGT, lw=1.8, label="目标 p（双峰，真实数据）", zorder=2)
    ax.plot(x, q_fwd, color=C_FWD, lw=2.4,
            label="前向 KL(p‖q)：最大似然 → 覆盖型", zorder=4)
    ax.plot(x, q_rev, color=C_REV, lw=2.4,
            label="反向 KL(q‖p)：寻峰型", zorder=3)

    ax.annotate("覆盖：铺满两峰，\n连中间空谷也占",
                xy=(0.0, gauss(0, mean_fwd, std_fwd)), xytext=(2.4, 0.30),
                color=C_FWD, fontsize=9.5, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_FWD, lw=1.0, alpha=0.85))
    ax.annotate("寻峰：锁定单峰，\n另一峰被彻底丢弃",
                xy=(-2.7, 0.42), xytext=(-7.8, 0.52),
                color=C_REV, fontsize=9.5, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_REV, lw=1.0, alpha=0.85))

    ax.grid(True, color=grid, alpha=0.3, linewidth=0.6)
    for sp in ax.spines.values():
        sp.set_color(grid)
    ax.tick_params(colors=fg, labelsize=9)
    ax.set_xlabel("x", color=fg)
    ax.set_ylabel("概率密度", color=fg)
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 0.72)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(fg)

    fig.tight_layout()
    fig.savefig(out, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    d = str(pathlib.Path(__file__).resolve().parent.parent / "docs" / "dit" / "images")
    os.makedirs(d, exist_ok=True)
    render(False, os.path.join(d, "kl-asymmetry-light.svg"))
    render(True, os.path.join(d, "kl-asymmetry-dark.svg"))
    print("前向 KL 最优单高斯:  mean=%.2f  std=%.3f  （矩匹配，覆盖双峰）" % (mean_fwd, std_fwd))
    print("反向 KL 最优单高斯:  mean=%.2f  std=%.3f  （锁定单峰）" % (mu_rev, std_rev))
    print("目标单峰自身 std = %.2f，可见反向 KL 的 std 更小（还会低估方差）" % SIG)
