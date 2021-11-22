job = 'source $HOME/.bashrc ; source activate threshold-devel ; python experiment.py --method {} --thresh {} --max-iters {} --num-burn {} --num-samples {} --num-steps-hyper {} --partial-momentum {} --check-prob 0.01 {} {} 2>/dev/null'

with open('joblist.txt', 'w') as f:
    for nb in [10000]:
        for ns in [100000]:
            for thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
                for num_steps_hyper in [6]:
                    for partial_momentum in [0.0]:
                        for nm in ['--newton-momentum', '--no-newton-momentum']:
                            nps = ['--no-newton-position'] if nm == '--no-newton-momentum' else ['--newton-position', '--no-newton-position']
                            for np in nps:
                                f.write(job.format('riemannian', thresh, 100, nb, ns, num_steps_hyper, partial_momentum, nm, np) + '\n')
            f.write(job.format('euclidean', 0.0, 0, nb, ns, 50, 0.0, '--no-newton-momentum', '--no-newton-position') + '\n')
