job = 'source $HOME/.bashrc ; source activate threshold-devel ; python experiment.py --method {} --thresh {} --max-iters {} --step-size {} --num-steps {} --num-samples {} --partial-momentum {} --check-prob 0.001 {} {} 2>/dev/null'
with open('joblist.txt', 'w') as f:
    for ns in [1000000]:
        for num_steps in [10]:
            for step_size in [0.1]:
                mi = 0
                thresh = 0.0
                f.write(job.format('euclidean', thresh, mi, step_size, num_steps, ns, 0.0, '--no-newton-momentum', '--no-newton-position') + '\n')

        for nm in ['--newton-momentum', '--no-newton-momentum']:
            nps = ['--no-newton-position'] if nm == '--no-newton-momentum' else ['--newton-position', '--no-newton-position']
            for np in nps:
                for num_steps in [25]:
                    for ps in [0.0]:
                        for step_size in [0.04]:
                            for thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
                                mi = 10000
                                f.write(job.format('riemannian', thresh, mi, step_size, num_steps, ns, ps, nm, np) + '\n')

        for num_steps in [25]:
            for ps in [0.0]:
                for step_size in [0.1]:
                    for thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
                        mi = 10000
                        f.write(job.format('implicit', thresh, mi, step_size, num_steps, ns, ps, '--no-newton-momentum', '--no-newton-position') + '\n')
