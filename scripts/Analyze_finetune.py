import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PALETTE = {"train": "#4472C4", "val": "#ED7D31", "test": "#70AD47"}

def load_data(train_data, train_label, val_data, val_label, test_data, test_label):
    print("[INFO] Loading Data...")
    if not os.path.exists(train_data):
        print(f"[ERROR] File not found: {train_data}")
        return None
    data = {
        "train": (np.load(train_data),  np.load(train_label)),
        "val":   (np.load(val_data),    np.load(val_label)),
        "test":  (np.load(test_data),   np.load(test_label)),
    }
    return data


def create_professional_dashboard(data, output_dir):
    print("[INFO] Rendering Signal Dashboard...")
    fig = plt.figure(figsize=(20, 11))
    fig.suptitle("DATASET FINETUNE OVERVIEW ", fontsize=26, fontweight='bold', y=0.96)

    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2], width_ratios=[1, 1.2, 1.2],
                           left=0.05, right=0.95, top=0.88, bottom=0.05, wspace=0.2, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [data['train'][0].shape[0], data['val'][0].shape[0], data['test'][0].shape[0]]
    ax1.pie(sizes, labels=['Train', 'Val', 'Test'], autopct='%1.1f%%',
            colors=[PALETTE['train'], PALETTE['val'], PALETTE['test']],
            explode=(0.05, 0, 0), textprops={'fontsize': 13, 'weight': 'bold'})
    ax1.set_title("(A) Dataset Split Ratio", fontsize=16, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1:])
    all_lbl = np.concatenate([data['train'][1], data['val'][1], data['test'][1]])
    classes = sorted(np.unique(all_lbl))
    x = np.arange(len(classes)); w = 0.25
    ax2.bar(x - w, [np.sum(data['train'][1] == c) for c in classes], w, label='Train', color=PALETTE['train'])
    ax2.bar(x,     [np.sum(data['val'][1]   == c) for c in classes], w, label='Val',   color=PALETTE['val'])
    ax2.bar(x + w, [np.sum(data['test'][1]  == c) for c in classes], w, label='Test',  color=PALETTE['test'])
    ax2.set_xticks(x); ax2.set_xticklabels([f"Class {c}" for c in classes], fontsize=12)
    ax2.set_title("(B) Class Balance", fontsize=16, fontweight='bold')
    ax2.legend(); ax2.grid(axis='y', alpha=0.3, linestyle='--')

    ax3 = fig.add_subplot(gs[1, 0])
    bp = ax3.boxplot([data['train'][0].flatten(), data['val'][0].flatten(), data['test'][0].flatten()],
                     tick_labels=['Train', 'Val', 'Test'], patch_artist=True)
    for p, c in zip(bp['boxes'], [PALETTE['train'], PALETTE['val'], PALETTE['test']]):
        p.set_facecolor(c); p.set_alpha(0.7)
    ax3.set_title("(C) Normalization Check", fontsize=16, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylabel("Values [0-1]")

    inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1:], wspace=0.15)
    ax4_bg = fig.add_subplot(gs[1, 1:]); ax4_bg.axis('off')
    ax4_bg.set_title("(D) Input Signal Visualization (Flattened Skeleton Vectors)",
                     fontsize=16, fontweight='bold', pad=25)

    unique_classes = np.unique(data['train'][1])[:3]
    class_names = {0: "Normal", 1: "Fall", 2: "Fight"}
    for i, cls in enumerate(unique_classes):
        ax = fig.add_subplot(inner_grid[0, i])
        idx = np.where(data['train'][1] == cls)[0][0]
        waveform = data['train'][0][idx].flatten()
        ax.plot(waveform, color=PALETTE['train'], linewidth=0.5, alpha=0.9)
        ax.fill_between(range(len(waveform)), waveform, color=PALETTE['train'], alpha=0.3)
        ax.set_ylim(0, 1); ax.set_xlim(0, len(waveform))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Sample #{idx}: {class_names.get(cls, f'Class {cls}')}",
                     fontsize=12, fontweight='bold', y=-0.15)
        for spine in ax.spines.values(): spine.set_edgecolor('#ccc')

    output_path = os.path.join(output_dir, "finetune1.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n DONE  {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze finetune dataset and save dashboard figures")
    parser.add_argument("--data_dir",   type=str,
                        default=os.path.join(_ROOT, "data", "finetune"),
                        help="Root finetune data directory (contains train/val/test subdirs)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_ROOT, "results", "figures", "data_analysis"),
                        help="Directory to save output figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_data(
        train_data=os.path.join(args.data_dir, "train", "train_data.npy"),
        train_label=os.path.join(args.data_dir, "train", "train_label.npy"),
        val_data=os.path.join(args.data_dir, "val",   "val_data.npy"),
        val_label=os.path.join(args.data_dir, "val",   "val_label.npy"),
        test_data=os.path.join(args.data_dir, "test",  "test_data.npy"),
        test_label=os.path.join(args.data_dir, "test",  "test_label.npy"),
    )
    if dataset:
        create_professional_dashboard(dataset, args.output_dir)