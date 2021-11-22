job = 'source $HOME/.bashrc ; source activate threshold-devel ; python experiment.py --method {} --thresh {} --max-iters {} --step-size {} --num-steps {} --num-samples {} --check-prob 0.01 {} {} 2>/dev/null'

with open('joblist.txt', 'w') as f:
    for ns in [100000]:
        mi = 0
        thresh = 0.0
        for step_size in [0.1, 0.2]:
            for num_steps in [20]:
                f.write(job.format('euclidean', thresh, mi, step_size, num_steps, ns, '--no-newton-momentum', '--no-newton-position') + '\n')

    step_sizes = [0.2]
    for ns in [100000]:
        for method in ['riemannian']:
            for step_size in step_sizes:
                for num_steps in [20]:
                    for nm in ['--newton-momentum', '--no-newton-momentum']:
                        nps = ['--no-newton-position'] if nm == '--no-newton-momentum' else ['--newton-position', '--no-newton-position']
                        for np in nps:
                            for thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
                                mi = 100
                                f.write(job.format(method, thresh, mi, step_size, num_steps, ns, nm, np) + '\n')
