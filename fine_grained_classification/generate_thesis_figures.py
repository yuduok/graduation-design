#!/usr/bin/env python3
"""
Generate thesis experiment figures
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Read experiment data
with open('output_fgd/oxford_pets/experiment_summary.json', 'r') as f:
    data = json.load(f)

# Create output directory for thesis figures
os.makedirs('thesis_figures', exist_ok=True)

# Configure matplotlib for Chinese support (use default font)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

# ==========================================
# Figure 1: Method Comparison Bar Chart
# ==========================================
methods = ['DynamicPromptTrainer', 'CoOp', 'CoCoOp']
shots = ['1', '2', '4', '8', '16']
epoch_config = data['epoch_config']

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(shots))
width = 0.25

for i, method in enumerate(methods):
    accs = []
    for shot in shots:
        shot_key = f"{shot}-shot"
        acc = data['results'][method][shot_key]['accuracy']
        accs.append(acc)
    
    offset = (i - 1) * width
    bars = ax.bar(x + offset, accs, width, 
                  label=method if method != 'DynamicPromptTrainer' else 'Ours (DynamicPrompt)',
                  alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Few-shot Setting', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax.set_title('Oxford-IIIT Pets Classification: Method Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f'{s}-shot' for s in shots])
ax.legend(loc='lower right')
ax.set_ylim(80, 92)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('thesis_figures/method_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: thesis_figures/method_comparison.png")

# ==========================================
# Figure 2: Adaptive Epoch Strategy
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
shot_list = [1, 2, 4, 8, 16]
epoch_list = [epoch_config[f'{s}-shot'] for s in shots]

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(shot_list)))
bars = ax.bar([f'{s}-shot' for s in shot_list], epoch_list, color=colors, alpha=0.8)

for bar, ep, s in zip(bars, epoch_list, shot_list):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'ep{ep}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Few-shot Setting', fontsize=12)
ax.set_ylabel('Number of Training Epochs', fontsize=12)
ax.set_title('Adaptive Epoch Strategy: Fewer Shots = More Epochs', fontsize=14)
ax.set_ylim(0, 120)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('thesis_figures/adaptive_epochs.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: thesis_figures/adaptive_epochs.png")

# ==========================================
# Figure 3: Comparison vs CoCoOp
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
comparison = data['comparison_vs_cocoop']

x = np.arange(len(shots))
differences = [comparison[f'{s}-shot'] for s in shots]
colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in differences]

bars = ax.bar([f'{s}-shot' for s in shots], differences, color=colors, alpha=0.8)

for bar, diff in zip(bars, differences):
    y_pos = diff + 0.05 if diff >= 0 else diff - 0.15
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{diff:+.1f}%',
            ha='center', va='bottom' if diff >= 0 else 'top', fontsize=10, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Few-shot Setting', fontsize=12)
ax.set_ylabel('Accuracy Difference vs CoCoOp (%)', fontsize=12)
ax.set_title('DynamicPromptTrainer vs CoCoOp: Performance Gap', fontsize=14)
ax.set_ylim(-2, 3.5)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('thesis_figures/comparison_vs_cocoop.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: thesis_figures/comparison_vs_cocoop.png")

# ==========================================
# Figure 4: All Methods Comparison (line chart)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

for method in methods:
    accs = []
    for shot in shots:
        shot_key = f"{shot}-shot"
        acc = data['results'][method][shot_key]['accuracy']
        accs.append(acc)
    
    label = method if method != 'DynamicPromptTrainer' else 'Ours (DynamicPrompt)'
    marker = 'o' if method == 'DynamicPromptTrainer' else 's' if method == 'CoOp' else '^'
    ax.plot([f'{s}-shot' for s in shots], accs, marker=marker, label=label, linewidth=2, markersize=8)

ax.set_xlabel('Few-shot Setting', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax.set_title('Oxford-IIIT Pets Classification: Method Comparison', fontsize=14)
ax.legend(loc='lower right')
ax.set_ylim(80, 92)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('thesis_figures/method_comparison_line.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: thesis_figures/method_comparison_line.png")

print("\nAll figures generated successfully!")
print("Figures saved in: thesis_figures/")
