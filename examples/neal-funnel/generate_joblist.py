job = 'source $HOME/.bashrc ; source activate threshold-devel ; python experiment.py --method {} --thresh {} --max-iters {} --step-size {} --num-steps {} --num-samples {} --check-prob {} --seed {} 2>/dev/null'
with open('joblist.txt', 'w') as f:
    for ns in [1000000]:
        for num_steps in [8]:
            for step_size in [0.001, 0.01, 0.1, 0.2]:
                mi = 0
                thresh = 0.0
                f.write(job.format('euclidean', thresh, mi, step_size, num_steps, ns, 0.001, 0) + '\n')

        for num_steps in [25]:
            for step_size in [0.2]:
                for thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
                    mi = 10000
                    f.write(job.format('softabs', thresh, mi, step_size, num_steps, ns, 0.001, 0) + '\n')

    for seed in range(1000):
        for ns in [10000]:
            for num_steps in [8]:
                for step_size in [0.1]:
                    mi = 0
                    thresh = 0.0
                    f.write(job.format('euclidean', thresh, mi, step_size, num_steps, ns, 0.0, seed) + '\n')

            for num_steps in [25]:
                for step_size in [0.2]:
                    for thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
                        mi = 10000
                        f.write(job.format('softabs', thresh, mi, step_size, num_steps, ns, 0.0, seed) + '\n')
