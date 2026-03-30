"""
实验结果汇总脚本
Collect experiment results and generate comparison tables/plots
"""
import os
import re
import json
import argparse


def find_experiment_dirs(base_dir):
    """查找所有实验目录"""
    experiments = []
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return experiments

    # base_dir 下的三个文件夹: CoCoOp, CoOp, DynamicPromptTrainer
    for trainer in os.listdir(base_dir):
        trainer_dir = os.path.join(base_dir, trainer)
        if not os.path.isdir(trainer_dir):
            continue
        
        # 解析 trainer 名称
        if trainer == "CoCoOp":
            trainer_name = "CoCoOp"
        elif trainer == "CoOp":
            trainer_name = "CoOp"
        elif trainer == "DynamicPromptTrainer":
            trainer_name = "DynamicPromptTrainer"
        else:
            continue
        
        # 遍历 shots 目录
        for shots_dir in os.listdir(trainer_dir):
            if not shots_dir.startswith("shots_"):
                continue
            shots = int(shots_dir.split("_")[1])
            full_path = os.path.join(trainer_dir, shots_dir)
            
            # 尝试从日志中解析训练的 epoch 数量
            epochs = None
            log_file = os.path.join(full_path, "log.txt")
            if os.path.exists(log_file):
                epochs = extract_epochs_from_log(log_file)
            
            # 查找 seed 子目录
            if os.path.isdir(full_path):
                for seed_dir in os.listdir(full_path):
                    if seed_dir.startswith("seed_"):
                        seed_path = os.path.join(full_path, seed_dir)
                        if os.path.isdir(seed_path):
                            # 尝试从子目录的日志获取 epoch
                            seed_log = os.path.join(seed_path, "log.txt")
                            if os.path.exists(seed_log):
                                epochs = extract_epochs_from_log(seed_log) or epochs
                            experiments.append({
                                "trainer": trainer_name,
                                "shots": shots,
                                "seed": seed_dir,
                                "epochs": epochs,
                                "path": seed_path,
                            })
            
            # 也检查直接在 shots 目录下的日志（兼容旧格式）
            if os.path.exists(log_file):
                # 检查是否已存在相同路径
                if not any(e["path"] == full_path for e in experiments):
                    experiments.append({
                        "trainer": trainer_name,
                        "shots": shots,
                        "seed": "default",
                        "epochs": epochs,
                        "path": full_path,
                    })
    
    return experiments


def extract_epochs_from_log(log_file):
    """从日志中解析训练的 epoch 总数"""
    try:
        with open(log_file, "r") as f:
            content = f.read()
            # 匹配 "epoch [50/50]" 或类似的格式
            match = re.search(r"epoch\s*\[(\d+)/(\d+)\]", content, re.IGNORECASE)
            if match:
                return int(match.group(2))  # 返回总 epoch 数
    except:
        pass
    return None


def find_latest_log(dir_path):
    """查找目录中最新/最完整的日志文件"""
    log_files = []
    if not os.path.isdir(dir_path):
        return None
    
    for f in os.listdir(dir_path):
        if f.startswith("log.txt"):
            full_path = os.path.join(dir_path, f)
            # 获取修改时间
            mtime = os.path.getmtime(full_path)
            log_files.append((mtime, full_path, f))
    
    if not log_files:
        return None
    
    # 按修改时间排序，返回最新的
    log_files.sort(key=lambda x: x[0], reverse=True)
    return log_files[0][1]


def extract_accuracy_from_log(log_dir_path):
    """从日志目录中提取最终测试准确率"""
    # 优先查找最新的日志文件
    log_file = find_latest_log(log_dir_path)
    
    if not log_file or not os.path.exists(log_file):
        return None

    best_acc = None
    final_test_acc = None

    with open(log_file, "r") as f:
        for line in f:
            # Dassl 格式: "accuracy: XX.X%"  或 "* accuracy: XX.X%"
            match = re.search(r"\*?\s*accuracy:\s*([\d.]+)%?", line, re.IGNORECASE)
            if match:
                acc = float(match.group(1))
                # 如果值小于1，说明是小数形式(0.xx)
                if acc < 1:
                    acc *= 100
                final_test_acc = acc

            # 也查找 "result" 字段
            match2 = re.search(r"result\s*[:=]\s*([\d.]+)", line, re.IGNORECASE)
            if match2:
                acc = float(match2.group(1))
                if acc < 1:
                    acc *= 100
                final_test_acc = acc

            # 查找 best result
            match3 = re.search(r"best_result\s*[:=]\s*([\d.]+)", line, re.IGNORECASE)
            if match3:
                acc = float(match3.group(1))
                if acc < 1:
                    acc *= 100
                best_acc = acc

    return best_acc if best_acc is not None else final_test_acc


def extract_training_curve(log_dir_path):
    """从日志目录中提取训练曲线数据"""
    # 查找最新的日志文件
    log_file = find_latest_log(log_dir_path)
    
    if not log_file or not os.path.exists(log_file):
        return []

    curve = []
    with open(log_file, "r") as f:
        for line in f:
            # 查找 epoch 和 accuracy/loss 信息
            epoch_match = re.search(r"epoch\s*\[?(\d+)", line, re.IGNORECASE)
            acc_match = re.search(r"acc[uracy]*\s*[:=]\s*([\d.]+)", line, re.IGNORECASE)
            loss_match = re.search(r"loss\s*[:=]\s*([\d.]+)", line, re.IGNORECASE)

            if epoch_match and (acc_match or loss_match):
                entry = {"epoch": int(epoch_match.group(1))}
                if acc_match:
                    acc = float(acc_match.group(1))
                    entry["acc"] = acc if acc > 1 else acc * 100
                if loss_match:
                    entry["loss"] = float(loss_match.group(1))
                curve.append(entry)
    return curve


def collect_results(base_dir):
    """收集所有实验结果"""
    experiments = find_experiment_dirs(base_dir)

    if not experiments:
        print("No experiments found. Run training first:")
        print("  bash run_experiments.sh cuda")
        return {}

    results = {}
    for exp in experiments:
        # 直接传入目录路径，让函数自己找最新日志
        acc = extract_accuracy_from_log(exp["path"])

        trainer = exp["trainer"]
        shots = exp["shots"]
        epochs = exp.get("epochs")

        if trainer not in results:
            results[trainer] = {}
        results[trainer][shots] = {
            "accuracy": acc,
            "path": exp["path"],
            "epochs": epochs,
        }

        status = f"{acc:.1f}%" if acc is not None else "N/A"
        epochs_str = f"(ep{epochs})" if epochs else ""
        print(f"  {trainer:25s} | {shots:2d}-shot {epochs_str:>8s} | {status}")

    return results


def filter_and_group_results(results):
    """过滤和整理结果，只保留 1, 4, 16, 32 shot 的 CoOp, CoCoOp, DynamicPromptTrainer"""
    filtered = {}
    
    # 只保留这三个模型
    valid_trainers = {"CoOp", "CoCoOp", "DynamicPromptTrainer"}
    valid_shots = {1, 4, 16, 32}
    
    for trainer, shots_data in results.items():
        if trainer not in valid_trainers:
            continue
        
        if trainer not in filtered:
            filtered[trainer] = {}
        
        for shots, data in shots_data.items():
            if shots in valid_shots:
                filtered[trainer][shots] = data
    
    return filtered


def collect_results_by_epochs(base_dir):
    """收集所有实验结果，按 epoch 分组"""
    experiments = find_experiment_dirs(base_dir)

    if not experiments:
        return {}

    # 按 trainer -> epochs -> shots 组织
    results = {}
    for exp in experiments:
        acc = extract_accuracy_from_log(exp["path"])

        trainer = exp["trainer"]
        shots = exp["shots"]
        epochs = exp.get("epochs")  # 可能为 None

        if trainer not in results:
            results[trainer] = {}
        
        # 用 epochs 作为 key
        epochs_key = epochs if epochs else "unknown"
        
        if epochs_key not in results[trainer]:
            results[trainer][epochs_key] = {}
        
        results[trainer][epochs_key][shots] = {
            "accuracy": acc,
            "path": exp["path"],
        }

        status = f"{acc:.1f}%" if acc is not None else "N/A"
        epochs_str = f"(ep{epochs})" if epochs else ""
        print(f"  {trainer:25s} | {shots:2d}-shot {epochs_str:>8s} | {status}")

    return results


def print_epoch_comparison_table(results):
    """打印按 epoch 分组的对比表格"""
    print("\n" + "=" * 80)
    print("  Comparison Table by Epochs: Top-1 Accuracy (%)")
    print("=" * 80)

    # 获取所有 trainer 和 epochs
    all_epochs = set()
    for trainer, epochs_data in results.items():
        all_epochs.update(epochs_data.keys())
    
    valid_epochs = sorted([e for e in all_epochs if e != "unknown"], key=lambda x: (x is None, x))
    
    for epochs in valid_epochs:
        print(f"\n--- Epochs: {epochs} ---")
        
        # 过滤这个 epoch 的数据
        filtered = {}
        for trainer, epochs_data in results.items():
            if epochs in epochs_data:
                filtered[trainer] = epochs_data[epochs]
        
        if not filtered:
            continue
            
        shots_list = sorted(set(s for t in filtered.values() for s in t.keys()))
        
        header = f"{'Method':<25s}"
        for s in shots_list:
            header += f" | {s:>2d}-shot"
        print(header)
        print("-" * 60)

        for trainer in sorted(filtered.keys()):
            display_name = "Ours" if trainer == "DynamicPromptTrainer" else trainer
            row = f"{display_name:<25s}"
            for s in shots_list:
                acc = filtered[trainer].get(s, {}).get("accuracy")
                if acc is not None:
                    row += f" | {acc:>6.1f}%"
                else:
                    row += f" |    N/A"
            print(row)

    print("=" * 80)


def print_comparison_table(results):
    """打印对比表格"""
    results = filter_and_group_results(results)
    
    trainers = sorted(results.keys())
    shots_list = sorted(
        set(s for t in results.values() for s in t.keys())
    )

    if not trainers:
        print("No results to display.")
        return

    # 终端表格
    print("\n" + "=" * 60)
    print("  Comparison Table: Top-1 Accuracy (%)")
    print("=" * 60)

    header = f"{'Method':<25s}"
    for s in shots_list:
        header += f" | {s:>2d}-shot"
    print(header)
    print("-" * 60)

    # 各方法结果
    for trainer in trainers:
        display_name = trainer
        if trainer == "DynamicPromptTrainer":
            display_name = "Ours"
        row = f"{display_name:<25s}"
        for s in shots_list:
            acc = results[trainer].get(s, {}).get("accuracy")
            if acc is not None:
                row += f" | {acc:>6.1f}%"
            else:
                row += f" |    N/A"
        print(row)

    print("=" * 60)


def print_latex_table(results):
    """生成 LaTeX 格式表格"""
    results = filter_and_group_results(results)
    
    trainers = sorted(results.keys())
    shots_list = sorted(
        set(s for t in results.values() for s in t.keys())
    )

    if not trainers:
        return

    print("\n% LaTeX Table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Oxford-IIIT Pets 分类准确率对比 (\\%)}")

    cols = "l" + "c" * len(shots_list)
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")

    header = "Method"
    for s in shots_list:
        header += f" & {s}-shot"
    header += " \\\\"
    print(header)
    print("\\midrule")

    # 各方法
    for trainer in trainers:
        display_name = trainer
        if trainer == "DynamicPromptTrainer":
            display_name = "\\textbf{Ours}"
        elif trainer == "CoOp":
            display_name = "CoOp"
        elif trainer == "CoCoOp":
            display_name = "CoCoOp"

        row = display_name
        for s in shots_list:
            acc = results[trainer].get(s, {}).get("accuracy")
            if acc is not None:
                # 加粗最高准确率
                row += f" & {acc:.1f}"
            else:
                row += " & -"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def plot_learning_curves(base_dir, results):
    """绘制学习曲线对比图"""
    results = filter_and_group_results(results)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    shots_list = sorted(
        set(s for t in results.values() for s in t.keys())
    )
    trainers = sorted(results.keys())
    colors = {
        "DynamicPromptTrainer": "#e74c3c",
        "CoOp": "#3498db",
        "CoCoOp": "#2ecc71",
    }
    labels = {
        "DynamicPromptTrainer": "Ours",
        "CoOp": "CoOp",
        "CoCoOp": "CoCoOp",
    }

    # 1. Few-shot 准确率对比柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = range(len(shots_list))
    width = 0.25

    for i, trainer in enumerate(trainers):
        accs = []
        for s in shots_list:
            acc = results[trainer].get(s, {}).get("accuracy", 0)
            accs.append(acc if acc else 0)

        offset = (i - len(trainers) / 2 + 0.5) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            accs,
            width,
            label=labels.get(trainer, trainer),
            color=colors.get(trainer, "#95a5a6"),
        )
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{acc:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xlabel("Few-shot Setting", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Fine-Grained Pet Classification: Method Comparison", fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{s}-shot" for s in shots_list])
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    output_path = os.path.join(base_dir, "comparison_bar.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nBar chart saved to: {output_path}")
    plt.close()

    # 2. 学习曲线（如果有训练日志数据）
    for shots in shots_list:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        for trainer in trainers:
            exp_data = results[trainer].get(shots, {})
            if not exp_data or not exp_data.get("path"):
                continue
            # 直接传入目录路径
            curve = extract_training_curve(exp_data["path"])
            if curve:
                epochs = [c["epoch"] for c in curve if "acc" in c]
                accs = [c["acc"] for c in curve if "acc" in c]
                if epochs:
                    ax.plot(
                        epochs,
                        accs,
                        label=labels.get(trainer, trainer),
                        color=colors.get(trainer, "#95a5a6"),
                        linewidth=2,
                    )
                    has_data = True

        if has_data:
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_title(f"Training Curves ({shots}-shot)", fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            output_path = os.path.join(base_dir, f"learning_curve_{shots}shot.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            print(f"Learning curve saved to: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_fgd", "oxford_pets"),
        help="Base directory for experiment outputs",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also output LaTeX table",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--epochs", "-e",
        action="store_true",
        help="Compare results by different epoch numbers",
    )
    args = parser.parse_args()

    print("Collecting experiment results...")
    print(f"Base directory: {args.base_dir}")
    print()

    if args.epochs:
        # 按 epoch 对比
        results = collect_results_by_epochs(args.base_dir)
        if results:
            print_epoch_comparison_table(results)
    else:
        results = collect_results(args.base_dir)
        if results:
            print_comparison_table(results)

            if args.latex:
                print_latex_table(results)

            if args.plot:
                plot_learning_curves(args.base_dir, results)

            # 保存结果到 JSON
            json_path = os.path.join(args.base_dir, "experiment_summary.json")
            filtered_results = filter_and_group_results(results)
            summary = {}
            for trainer, shots_data in filtered_results.items():
                summary[trainer] = {}
                for shots, data in shots_data.items():
                    summary[trainer][str(int(shots))] = data.get("accuracy")

            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nResults saved to: {json_path}")
        else:
            print("\nNo results found. Please run experiments first:")
            print("  bash run_experiments.sh cuda")


if __name__ == "__main__":
    main()
