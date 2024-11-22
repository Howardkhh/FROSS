import os
from tqdm import tqdm
import threading
import time
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Extract zip sequences from 3RScan dataset and preprocess it.')
parser.add_argument('--path', type=Path, required=True, help='3RScan directory')
parser.add_argument('--rio_renderer_path', type=Path, required=True, help='Path to rio_renderer build')
parser.add_argument('--nthreads', type=int, default=32, help='Number of threads to use')
args = parser.parse_args()

data_folder = args.path.resolve() / "data"
rio_renderer_path = args.rio_renderer_path
nthreads = args.nthreads
os.chdir(rio_renderer_path)
finished = []
errored = False
error_id = ""

def extract_boxes(dirs, tid, nthreads):
    global errored, error_id
    for i in range(tid, len(dirs), nthreads):
        dir = dirs[i]

        ret = os.system(f"unzip -q -o {os.path.join(data_folder, dir, 'sequence.zip')} -d {os.path.join(data_folder, dir, 'sequence')}")
        if ret != 0:
            errored = True
            error_id = dir
        if errored:
            return

        ret = os.system(f"./rio_renderer_render_all {data_folder} {dir} sequence 0 > /dev/null")
        if ret != 0:
            errored = True
            error_id = dir
        if errored:
            return
        
        finished[i] = 1

dirs = sorted(os.listdir(data_folder))
dirs = [dir for dir in dirs if not dir.endswith(".json")]
finished = [0] * len(dirs)
threads = []
for i in range(nthreads):
    threads.append(threading.Thread(target=extract_boxes, args=(dirs, i, nthreads)))
    threads[-1].start()

bar = tqdm(total=len(dirs))
while sum(finished) < len(dirs):
    if errored:
        break
    time.sleep(0.1)
    bar.update(sum(finished) - bar.n)
bar.close()

for i in range(nthreads):
    threads[i].join()
if errored:
    raise Exception(f"Error in {error_id}")