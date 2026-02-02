from pathlib import Path

RESULTS_DIR = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/results"
OUT_FILE = "jobs.txt"

jobs = []

for ct_dir in Path(RESULTS_DIR).iterdir():
    if not ct_dir.is_dir():
        continue
    for run_dir in ct_dir.iterdir():
        if run_dir.is_dir():
            jobs.append(f"{ct_dir.name} {run_dir.name}")

with open(OUT_FILE, "w") as f:
    f.write("\n".join(jobs))

print(f"Wrote {len(jobs)} jobs to {OUT_FILE}")
