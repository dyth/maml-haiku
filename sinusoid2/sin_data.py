import numpy as onp
from jax import numpy as np


def meta_train_data(tasks_per_batch, support_size, query_size):
    'generate support and query data for sine regression in MAML'
    # randomly select amplitude and phase for each task
    As     = []
    phases = []
    for _ in range(tasks_per_batch):
        As.append(onp.random.uniform(low=0.1, high=5.))
        phases.append(onp.random.uniform(low=0, high=np.pi))
    def get_batch(size):
        'for each task randomly sample input space and create according output'
        xs, ys = [], []
        for A, phase in zip(As, phases):
            x = onp.random.uniform(low=-5., high=5., size=(size, 1))
            y = A * onp.sin(x + phase)
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)
    support = get_batch(support_size)
    query   = get_batch(query_size)
    return support, query


def training_data(size):
    x = onp.random.uniform(low=-5., high=5., size=(size, 1))
    y = 1. * onp.sin(x + np.pi/4)
    return x, y

def testing_data():
    x = np.linspace(-5, 5, 100).reshape((100, 1))
    y = 1. * onp.sin(x + np.pi/4)
    return x, y
