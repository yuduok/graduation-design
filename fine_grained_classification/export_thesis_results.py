"""
为论文导出实验结果的脚本。

作用：
1. 从 output_fgd/oxford_pets/experiment_summary.json 提取主实验结果
2. 如果存在 security evaluation 输出，则一并整理
3. 生成 thesis/generated_results.json 供论文写作引用
"""
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
SUMMARY_FILE = ROOT / "output_fgd" / "oxford_pets" / "experiment_summary.json"
OUTPUT_FILE = ROOT / "thesis" / "generated_results.json"


def load_main_results():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_FILE}")
    return json.loads(SUMMARY_FILE.read_text())


def load_security_results():
    candidates = list((ROOT / "security_results").glob("security_evaluation_*.txt"))
    parsed = {}
    pattern = re.compile(r"^(.*?):\s*([+-]?\d+(?:\.\d+)?)%?\s*$")
    for path in candidates:
        data = {}
        for line in path.read_text().splitlines():
            m = pattern.match(line.strip())
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                data[key] = float(m.group(2))
        if data:
            parsed[path.stem] = data
    return parsed


def main():
    main_results = load_main_results()
    payload = {
        "main_experiments": main_results,
        "security_experiments": load_security_results(),
        "notes": {
            "main_source": str(SUMMARY_FILE.relative_to(ROOT)),
            "security_source_dir": "security_results",
        },
    }
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved thesis results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
