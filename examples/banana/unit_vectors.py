import os
import pickle

import numpy as np


def generate_unit_vectors():
    U = np.random.normal(size=(10000, 2))
    U = U / np.linalg.norm(U, axis=-1, keepdims=True)
    return U

def load_unit_vectors():
    with open(os.path.join('data', 'unit-vectors.pkl'), 'rb') as f:
        U = pickle.load(f)['U']
    return U

def main():
    U = generate_unit_vectors()
    with open(os.path.join('data', 'unit-vectors.pkl'), 'wb') as f:
        pickle.dump({'U': U}, f)

if __name__ == '__main__':
    main()
