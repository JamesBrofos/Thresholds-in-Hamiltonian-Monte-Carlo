import glob
import os


job = 'source $HOME/.bashrc ; source activate threshold-devel ; python compute_metrics.py --file-name {}'
fns = glob.glob(os.path.join('samples', '*num-samples-1000000*.pkl'))
with open('metricslist.txt', 'w') as f:
    for fn in fns:
        f.write(job.format(fn) + '\n')

