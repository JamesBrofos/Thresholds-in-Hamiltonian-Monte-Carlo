job = 'source $HOME/.bashrc ; source activate threshold-devel ; python experiment.py --method {} --thresh {} --max-iters {} --step-size {} --num-steps {} --num-samples {} --partial-momentum {} --check-prob 0.01 --{} {} {} 2>/dev/null'
with open('joblist.txt', 'w') as f:
    for ns in [100000]:
        for num_steps in [10]:
            for step_size in [0.015]:
                mi = 0
                thresh = 0.0
                f.write(job.format('euclidean', thresh, mi, step_size, num_steps, ns, 0.0, 'correct', '--no-newton-momentum', '--no-newton-position') + '\n')

        for num_steps in [6]:
            for partial_momentum in [0.0]:
                for step_size in [0.5]:
                    for thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
                        for c in ['correct', 'no-correct']:
                            mi = 10000
                            if c == 'correct':
                                for nm in ['--newton-momentum', '--no-newton-momentum']:
                                    nps = ['--no-newton-position'] if nm == '--no-newton-momentum' else ['--newton-position', '--no-newton-position']
                                    for np in nps:
                                        f.write(job.format('riemannian', thresh, mi, step_size, num_steps, ns, partial_momentum, c, nm, np) + '\n')
                            else:
                                f.write(job.format('riemannian', thresh, mi, step_size, num_steps, ns, partial_momentum, c, '--no-newton-momentum', '--no-newton-position') + '\n')
