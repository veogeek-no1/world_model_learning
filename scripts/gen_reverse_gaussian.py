"""验证"步小 → 反向条件近似高斯"：双点数据下的精确反向条件分布。

数据分布取 p_data = ½δ₋₁ + ½δ₊₁（只有 ±1 两个点）。此时反向条件
q(x_{t-1}|x_t) 可用 DDPM 后验公式精确求出——它是两个高斯的混合：
    q(x_{t-1}|x_t) = Σ_k w_k(x_t)·N(μ̃(x_t,k), β̃_t),  k∈{−1,+1}
两分量均值间距 ~O(β_t)，分量宽度 ~O(√β_t)，比值 ~O(√β_t)：
步子小 → 两峰融成单高斯；步子大 → 明显双峰，单高斯拟合失效。

两个 panel 共享同一噪声水平 ᾱ_t = 0.5，只改单步大小 β_t。
输出 docs/dit/images/reverse-gaussian-{light,dark}.svg。
重新生成：python scripts/gen_reverse_gaussian.py
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


def gauss(x, mu, sig2):
    return np.exp(-0.5 * (x - mu) ** 2 / sig2) / np.sqrt(2 * np.pi * sig2)


ABAR_T = 0.5     # 固定当前时刻的总噪声水平
X_T = 0.15       # 固定观测到的 x_t（略偏 +1 侧，让两分量权重不对称）


def exact_posterior(beta, x):
    """双点数据下 q(x_{t-1}|x_t) 的精确形式（DDPM 后验公式逐项代入）。"""
    alpha = 1.0 - beta
    abar_prev = ABAR_T / alpha                       # ᾱ_{t-1} = ᾱ_t / α_t
    assert abar_prev <= 1.0, "β 太大：ᾱ_{t-1} 会超过 1"
    coef_x0 = np.sqrt(abar_prev) * beta / (1 - ABAR_T)
    coef_xt = np.sqrt(alpha) * (1 - abar_prev) / (1 - ABAR_T)
    beta_tilde = (1 - abar_prev) / (1 - ABAR_T) * beta

    ks = np.array([-1.0, 1.0])
    w = gauss(X_T, np.sqrt(ABAR_T) * ks, 1 - ABAR_T)  # 每个数据点的后验责任
    w = w / w.sum()
    mus = coef_x0 * ks + coef_xt * X_T

    pdf = sum(wk * gauss(x, mk, beta_tilde) for wk, mk in zip(w, mus))
    # 矩匹配的单高斯拟合（如果硬用一个高斯去近似它）
    m = np.sum(w * mus)
    v = np.sum(w * (beta_tilde + mus ** 2)) - m ** 2
    fit = gauss(x, m, v)

    sep = mus[1] - mus[0]
    width = np.sqrt(beta_tilde)
    dx = x[1] - x[0]
    eps = 1e-300
    kl = np.sum(pdf * np.log((pdf + eps) / (fit + eps))) * dx
    return pdf, fit, dict(sep=sep, width=width, ratio=sep / width, kl=kl,
                          mus=mus, w=w, beta_tilde=beta_tilde)


BETA_SMALL, BETA_LARGE = 0.02, 0.4
x_s = np.linspace(-0.6, 0.9, 3000)
x_l = np.linspace(-2.2, 2.2, 3000)
pdf_s, fit_s, st_s = exact_posterior(BETA_SMALL, x_s)
pdf_l, fit_l, st_l = exact_posterior(BETA_LARGE, x_l)

C_EXACT, C_FIT = "#4c6ef5", "#e8710a"


def render(dark: bool, out: str) -> None:
    fg = "#c9d1d9" if dark else "#24292f"
    grid = "#8b949e" if dark else "#d0d7de"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_alpha(0)

    for ax, x, pdf, fit, st, beta, verdict in (
        (axes[0], x_s, pdf_s, fit_s, st_s, BETA_SMALL, "两分量重叠 → 单高斯拟合几乎完美"),
        (axes[1], x_l, pdf_l, fit_l, st_l, BETA_LARGE, "明显双峰 → 单高斯拟合失效"),
    ):
        ax.patch.set_alpha(0)
        ax.plot(x, pdf, color=C_EXACT, lw=2.4, label="精确的 q(x_{t-1}|x_t)")
        ax.plot(x, fit, color=C_FIT, lw=2.0, ls="--", label="硬用单高斯拟合（矩匹配）")
        ax.set_title(f"β_t = {beta}：{verdict}", fontsize=10.5, color=fg)
        ax.set_xlabel("$x_{t-1}$", color=fg)
        ax.grid(True, color=grid, alpha=0.3, linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color(grid)
        ax.tick_params(colors=fg, labelsize=9)
        txt = (f"分量间距 / 宽度 = {st['ratio']:.2f}\n"
               f"KL(精确‖拟合) = {st['kl']:.1e}")
        ax.text(0.03, 0.95, txt, transform=ax.transAxes, va="top",
                fontsize=9, color=fg)
        leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        for t in leg.get_texts():
            t.set_color(fg)

    axes[0].set_ylabel("概率密度", color=fg)
    fig.suptitle("同一噪声水平（ᾱ_t = 0.5）、同一观测 x_t，只改单步大小 β_t",
                 fontsize=11, color=fg, y=1.0)
    fig.tight_layout()
    fig.savefig(out, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    d = str(pathlib.Path(__file__).resolve().parent.parent / "docs" / "dit" / "images")
    os.makedirs(d, exist_ok=True)
    render(False, os.path.join(d, "reverse-gaussian-light.svg"))
    render(True, os.path.join(d, "reverse-gaussian-dark.svg"))
    for name, st, beta in (("小步", st_s, BETA_SMALL), ("大步", st_l, BETA_LARGE)):
        print(f"{name} β={beta}: 间距={st['sep']:.4f}  宽度={st['width']:.4f}  "
              f"间距/宽度={st['ratio']:.2f}  KL(精确‖单高斯)={st['kl']:.5f}  "
              f"权重={np.round(st['w'],3)}")
