job = 'source $HOME/.bashrc ; source activate threshold-devel ; python experiment.py --method {} --thresh {} --max-iters {} --num-burn {} --num-samples {} --num-steps-hyper {} --step-size-hyper {} --num-steps-gaussian {} --step-size-gaussian {} --check-prob 0.01 {} {} 2>/dev/null'

params = {
    32: {
        'riemannian': {
            'num_steps_hyper': 5,
            'step_size_hyper': 0.2,
            'num_steps_gaussian': 50,
            'step_size_gaussian': 0.3
        },
        'euclidean': {
            'num_steps_hyper': 5,
            'step_size_hyper': 0.05,
            'num_steps_gaussian': 50,
            'step_size_gaussian': 0.05
        }
    },
    16: {
        'riemannian': {
            'num_steps_hyper': 5,
            'step_size_hyper': 0.1,
            'num_steps_gaussian': 50,
            'step_size_gaussian': 0.2
        },
        'euclidean': {
            'num_steps_hyper': 10,
            'step_size_hyper': 0.05,
            'num_steps_gaussian': 50,
            'step_size_gaussian': 0.15
        }
    }
}

with open('joblist.txt', 'w') as f:
    for ng in [32]:
        if ng == 16:
            nb = 10000
            ns = 100000
        elif ng == 32:
            nb = 1000
            ns = 5000

        for thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
            for num_steps_hyper in [5]:
                for nm in ['--newton-momentum', '--no-newton-momentum']:
                    nps = ['--no-newton-position'] if nm == '--no-newton-momentum' else ['--newton-position', '--no-newton-position']
                    for np in nps:
                        p = params[ng]['riemannian']
                        f.write(job.format('riemannian', thresh, 100, nb, ns, p['num_steps_hyper'], p['step_size_hyper'], p['num_steps_gaussian'], p['step_size_gaussian'], nm, np) + '\n')
        p = params[ng]['euclidean']
        f.write(job.format('euclidean', 0.0, 0, nb, ns, p['num_steps_hyper'], p['step_size_hyper'], p['num_steps_gaussian'], p['step_size_gaussian'], '--no-newton-momentum', '--no-newton-position') + '\n')
