"""
With thanks to Eric Jang from
https://jax.readthedocs.io/en/stable/notebooks/maml.html
"""
import numpy as onp
from functools import partial # for use with vmap
from matplotlib import pyplot as plt

from jax import numpy as np
from jax import vmap, jit, random, grad
from jax.experimental import optimizers
# Element-wise manipulation of collections of numpy arrays
from jax.tree_util import tree_multimap


from model import create_model

rng = random.PRNGKey(0)
alpha = .1
support_size = 20
query_size = 1
meta_training_epochs = 20000
tasks_per_batch = 4


def sample_tasks(tasks_per_batch, support_size, query_size):
    # Select amplitude and phase for the task
    As = []
    phases = []
    for _ in range(tasks_per_batch):
        As.append(onp.random.uniform(low=0.1, high=.5))
        phases.append(onp.random.uniform(low=0., high=np.pi))
    def get_batch(size):
        xs, ys = [], []
        for A, phase in zip(As, phases):
            x = onp.random.uniform(low=-5., high=5., size=(size, 1))
            y = A * onp.sin(x + phase)
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)
    x1, y1 = get_batch(support_size)
    x2, y2 = get_batch(query_size)
    return x1, y1, x2, y2

def loss(params, inputs, targets):
    'compute average loss for the batch'
    predictions = net_apply(params, inputs)
    return np.mean((targets - predictions)**2)

def inner_update(p, x1, y1):
    grads = grad(loss)(p, x1, y1)
    inner_sgd_fn = lambda g, state: (state - alpha*g)
    return tree_multimap(inner_sgd_fn, grads, p)

def maml_loss(p, x1, y1, x2, y2):
    p2 = inner_update(p, x1, y1)
    return loss(p2, x2, y2)

def batch_maml_loss(p, x1_b, y1_b, x2_b, y2_b):
    'vmapped version of maml loss, return scalar for all tasks'
    task_losses = vmap(partial(maml_loss, p))(x1_b, y1_b, x2_b, y2_b)
    return np.mean(task_losses)

@jit
def step(i, opt_state, x1, y1, x2, y2):
    p = get_params(opt_state)
    g = grad(batch_maml_loss)(p, x1, y1, x2, y2)
    l = batch_maml_loss(p, x1, y1, x2, y2)
    return opt_update(i, g, opt_state), l


# initialise model and optimiser
net_apply, net_params = create_model(rng)
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
opt_state = opt_init(net_params)

# meta-train
for i in range(meta_training_epochs):
    x1, y1, x2, y2 = sample_tasks(tasks_per_batch, support_size, query_size)
    opt_state, l = step(i, opt_state, x1, y1, x2, y2)
    if i % 1000 == 0:
        print(i)

# training data
xrange_inputs = np.linspace(-5, 5, 100).reshape((100, 1))
targets = 1. * onp.sin(xrange_inputs + 0.)
# testing data
x1 = onp.random.uniform(low=-5., high=5., size=(support_size, 1))
y1 = 1. * onp.sin(x1 + 0.)

# meta-testing
net_params = get_params(opt_state)
preds = [vmap(partial(net_apply, net_params))(xrange_inputs)] # zero-shot generalisation
for i in range(1, 5):
    net_params = inner_update(net_params, x1, y1) # training
    preds.append(vmap(partial(net_apply, net_params))(xrange_inputs)) # testing

# plot
plt.plot(xrange_inputs, targets, label='target')
for i, p in enumerate(preds):
    plt.plot(xrange_inputs, p, label=f'{i}-shot')
plt.legend()
plt.show()
