"""生成 noise schedule 对比图（linear vs cosine），明暗各一版。

输出到 docs/dit/images/，供 diffusion-basics.md 用 #only-light / #only-dark 引用。

两个要点：
- svg.fonttype='path' 把文字转成矢量路径，读者没装中文字体也能正确显示。
- cosine schedule 按 Nichol & Dhariwal 2021 的做法把 beta 截断在 0.999，
  否则 abar_T 会精确塌到 0，末步方差退化。
"""
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- 中文字体：转 path 后不依赖读者环境 ---
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

T = 1000
t = np.arange(1, T + 1)

# --- linear schedule (DDPM, Ho et al. 2020): beta 1e-4 -> 0.02 ---
beta_lin = np.linspace(1e-4, 0.02, T)
abar_lin = np.cumprod(1.0 - beta_lin)

# --- cosine schedule (Improved DDPM, Nichol & Dhariwal 2021) ---
s = 0.008
ts = np.arange(0, T + 1) / T
f = np.cos((ts + s) / (1 + s) * np.pi / 2) ** 2
abar_raw = f / f[0]
beta_cos = np.clip(1.0 - abar_raw[1:] / abar_raw[:-1], 0.0, 0.999)  # 论文的 0.999 截断
abar_cos = np.cumprod(1.0 - beta_cos)

eps = 1e-12
snr_lin = np.log(abar_lin / (1 - abar_lin + eps) + eps)
snr_cos = np.log(abar_cos / (1 - abar_cos + eps) + eps)

C_LIN, C_COS = "#e8710a", "#4c6ef5"  # 橙 / 蓝：明暗背景都可读，对色觉障碍友好


def render(dark: bool, out: str) -> None:
    fg = "#c9d1d9" if dark else "#24292f"
    grid = "#8b949e" if dark else "#d0d7de"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    fig.patch.set_alpha(0)

    for ax in axes:
        ax.patch.set_alpha(0)
        ax.grid(True, color=grid, alpha=0.35, linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color(grid)
        ax.tick_params(colors=fg, labelsize=9)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)

    ax = axes[0]
    ax.plot(t, abar_lin, color=C_LIN, lw=2.2, label="linear (DDPM 2020)")
    ax.plot(t, abar_cos, color=C_COS, lw=2.2, label="cosine (iDDPM 2021)")
    ax.set_xlabel("时间步 $t$")
    ax.set_ylabel(r"$\bar{\alpha}_t$")
    ax.set_title(r"信号残留比例 $\bar{\alpha}_t$ 的衰减", fontsize=11)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.02, 1.02)
    leg = ax.legend(frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(fg)

    i50 = int(np.argmax(abar_lin < 0.5))
    ax.axvline(i50, color=C_LIN, ls=":", lw=1.2, alpha=0.7)
    ax.annotate(
        f"linear 在 t≈{i50} 就丢掉一半信号",
        xy=(i50, 0.5), xytext=(i50 + 80, 0.74),
        color=fg, fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=fg, lw=0.9, alpha=0.8),
    )

    ax = axes[1]
    ax.plot(t, snr_lin, color=C_LIN, lw=2.2, label="linear")
    ax.plot(t, snr_cos, color=C_COS, lw=2.2, label="cosine")
    ax.axhline(0, color=grid, lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel("时间步 $t$")
    ax.set_ylabel(r"$\log \mathrm{SNR}_t$")
    ax.set_title("对数信噪比（0 以下：噪声已压过信号）", fontsize=11)
    ax.set_xlim(0, T)
    leg = ax.legend(frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(fg)

    fig.tight_layout()
    fig.savefig(out, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    d = str(pathlib.Path(__file__).resolve().parent.parent / "docs" / "dit" / "images")
    os.makedirs(d, exist_ok=True)
    render(False, os.path.join(d, "noise-schedule-light.svg"))
    render(True, os.path.join(d, "noise-schedule-dark.svg"))

    # 正文引用的数字，必须与图一致
    print("linear  abar_T = %.3e   | cosine  abar_T = %.3e" % (abar_lin[-1], abar_cos[-1]))
    print("跌破 0.5:  linear t=%d  | cosine t=%d" % (
        int(np.argmax(abar_lin < 0.5)), int(np.argmax(abar_cos < 0.5))))
    print("末 100 步 abar 均值: linear %.2e | cosine %.2e" % (
        abar_lin[-100:].mean(), abar_cos[-100:].mean()))
    print("beta_cos 被截断到 0.999 的步数:", int((beta_cos >= 0.999).sum()))
