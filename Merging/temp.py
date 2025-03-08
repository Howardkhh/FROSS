import os
from pathlib import Path

results_dir = "output/grid_search_kf/results_wrong/replica"
new_results_dir = "output/grid_search_kf/results/replica"
results_file = [f for f in os.listdir(results_dir) if not f.endswith('gtpose.txt')]

for file in results_file:
    with open(os.path.join(results_dir, file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('KF ratio: '):
                break
    file = Path(file).stem + '_gtpose.txt'
    with open(os.path.join(new_results_dir, file), 'w') as f:
        f.write(line)