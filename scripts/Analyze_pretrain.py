import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ===================== CẤU HÌNH =====================
# Có thể truyền path qua argument: python Analyze_pretrain.py --train_path ... --val_path ...
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JOINT_NAMES = [
    "Nose","L.Eye","R.Eye","L.Ear","R.Ear",
    "L.Shldr","R.Shldr","L.Elbow","R.Elbow",
    "L.Wrist","R.Wrist","L.Hip","R.Hip",
    "L.Knee","R.Knee","L.Ankle","R.Ankle","Neck"
]
# ====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

C_TRAIN = "#2563EB"
C_VAL   = "#F59E0B"
C_BG    = "#F8FAFC"
C_TITLE = "#1E293B"
C_GRID  = "#E2E8F0"


def load_and_sample(path, n_sample=2000):
    data  = np.load(path, mmap_mode='r')
    total = data.shape[0]
    N, C, T, V, M = data.shape

    idx    = np.random.choice(total, min(total, n_sample), replace=False)
    subset = np.array(data[idx])

    # Active values: Person-0 only
    flat_p0 = subset[:, :, :, :, 0].flatten()
    active  = flat_p0[np.abs(flat_p0) > 1e-5]

    # Joint variance: std theo T → (n,C,V,M), mean theo (n,C,M) → (V,)
    joint_var = np.mean(np.std(subset, axis=2), axis=(0, 1, 3))

    # Temporal activity: mean |velocity| theo thời gian
    p0_x  = subset[:, 0, :, :, 0]                      # (n, T, V)
    vel   = np.abs(np.diff(p0_x, axis=1))               # (n, T-1, V)
    mask  = np.abs(p0_x[:, :-1, :]) > 1e-5
    temporal = np.nanmean(np.where(mask, vel, np.nan), axis=(0, 2))  # (T-1,)

    return {
        "total":    total,
        "shape":    (C, T, V, M),
        "active":   active,
        "joint_var": joint_var,
        "temporal": temporal,
        "T":        T,
    }


def draw_dashboard(tr, va, save_path):
    plt.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlepad":     10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   10,
        "font.family":       "DejaVu Sans",
    })

    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    fig.suptitle("DATASET PRETRAIN OVERVIEW",
                 fontsize=20, fontweight="bold", color=C_TITLE, y=0.97)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[1, 1.6],
                           hspace=0.42, wspace=0.30,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    # A: Dataset Split Ratio 
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(C_BG)
    total_all = tr["total"] + va["total"]
    sizes = [tr["total"] / total_all * 100, va["total"] / total_all * 100]
    wedges, texts, autotexts = ax_a.pie(
        sizes, labels=["Train", "Val"], colors=[C_TRAIN, C_VAL],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11),
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold")
    ax_a.set_title("(A) Dataset Split Ratio")
    C_dim, T, V, M = tr["shape"]
    info = (f"Train : {tr['total']:,} samples\n"
            f"Val      : {va['total']:,} samples\n"
            f"Shape : C={C_dim}, T={T}, V={V}, M={M}")
    ax_a.text(0.5, -0.14, info, transform=ax_a.transAxes,
              ha="center", fontsize=9, color="#475569",
              bbox=dict(facecolor="white", edgecolor=C_GRID, boxstyle="round,pad=0.5"))

    # B: Joint Variance 
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(C_BG)
    n_joints = len(tr["joint_var"])
    x_j      = np.arange(n_joints)
    jnames   = JOINT_NAMES[:n_joints] if n_joints <= len(JOINT_NAMES) \
               else [str(i) for i in range(n_joints)]
    norm   = plt.Normalize(tr["joint_var"].min(), tr["joint_var"].max())
    colors = plt.cm.RdYlGn(norm(tr["joint_var"]))
    bar_w  = 0.4
    ax_b.bar(x_j - bar_w/2, tr["joint_var"], width=bar_w,
             color=colors, edgecolor="white", linewidth=0.6, zorder=2)
    ax_b.bar(x_j + bar_w/2, va["joint_var"], width=bar_w,
             color=C_VAL, alpha=0.85, edgecolor="white", linewidth=0.6,
             zorder=2, label="Val")
    mean_v = tr["joint_var"].mean()
    ax_b.axhline(mean_v, color="#64748B", linewidth=1.2, linestyle="--",
                 label=f"Train mean = {mean_v:.3f}", zorder=3)
    ax_b.set_xticks(x_j)
    ax_b.set_xticklabels(jnames, rotation=45, ha="right", fontsize=8)
    ax_b.set_ylabel("Motion Intensity (Std Dev)")
    ax_b.set_ylim(bottom=0)
    ax_b.set_title("(B) Kinematic Diversity  –  Motion Variance per Joint")
    train_patch = mpatches.Patch(facecolor="#22c55e", edgecolor="white",
                                 label="Train")
    handles, _ = ax_b.get_legend_handles_labels()
    ax_b.legend(handles=[train_patch] + handles, loc="upper right",
                framealpha=0.9, fontsize=9)
    ax_b.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax_b.set_axisbelow(True)

    #  C: Normalization Check – Value Distribution
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(C_BG)
    rng  = np.random.default_rng(42)
    tr_s = rng.choice(tr["active"], size=min(50000, len(tr["active"])), replace=False)
    va_s = rng.choice(va["active"], size=min(50000, len(va["active"])), replace=False)
    sns.kdeplot(tr_s, fill=True, alpha=0.55, color="#C8B89A", ax=ax_c)
    sns.kdeplot(tr_s, fill=False, color=C_TRAIN, linewidth=2, label="Train", ax=ax_c)
    sns.kdeplot(va_s, fill=False, color=C_VAL, linewidth=2,
                linestyle="--", label="Val", ax=ax_c)
    ax_c.set_xlim(-1.2, 1.2)
    ax_c.axvline(0, color="#64748B", linewidth=1.2, linestyle=":")
    ax_c.set_xlabel("Normalized Coordinate Value")
    ax_c.set_ylabel("Density")
    ax_c.set_title("(C) Normalization Check")
    ax_c.legend(framealpha=0.9)
    ax_c.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax_c.set_axisbelow(True)

    #  D: Temporal Activity Pattern 
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(C_BG)
    t_tr = np.arange(len(tr["temporal"]))
    t_va = np.arange(len(va["temporal"]))


    ax_d.plot(t_tr, tr["temporal"], color=C_TRAIN, linewidth=2, label="Train", zorder=3)
    ax_d.plot(t_va, va["temporal"], color=C_VAL, linewidth=2,
              linestyle="--", label="Val", zorder=3)
    ax_d.set_xlabel("Time Frame")
    ax_d.set_ylabel("Mean Joint Velocity")
    ax_d.set_title("(D) Temporal Activity Pattern  –  Motion Across Time")
    ax_d.legend(framealpha=0.9, fontsize=9)
    ax_d.set_xlim(0, len(t_tr) - 1)
    ax_d.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax_d.set_axisbelow(True)

    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    print(f" Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pretrain dataset and save dashboard figures")
    parser.add_argument("--train_path", type=str,
                        default=os.path.join(_ROOT, "data", "pretrain", "train", "train_data.npy"),
                        help="Path to pretrain train_data.npy")
    parser.add_argument("--val_path",   type=str,
                        default=os.path.join(_ROOT, "data", "pretrain", "val", "val_data.npy"),
                        help="Path to pretrain val_data.npy")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_ROOT, "results", "figures", "data_analysis"),
                        help="Directory to save output figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(" Loading & sampling data...")
    tr = load_and_sample(args.train_path)
    va = load_and_sample(args.val_path)
    draw_dashboard(tr, va, os.path.join(args.output_dir, "pretrain.png"))
    print(" Done!")