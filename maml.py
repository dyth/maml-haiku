"""
With thanks to Eric Jang from
https://jax.readthedocs.io/en/stable/notebooks/maml.html
"""
from functools import partial # for use with vmap
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm

from jax import numpy as np
from jax import vmap, jit, random, grad
from jax.experimental import optimizers
# Element-wise manipulation of collections of numpy arrays
from jax.tree_util import tree_multimap

from model import create_model
from sin_data import meta_train_data, training_data, testing_data


# hyperparameters from https://github.com/cbfinn/maml/blob/master/main.py
rng = random.PRNGKey(0)
metatrain_iterations = 7000
meta_lr = 0.001
update_lr = 0.01
support_size = 10
query_size = 10
meta_batch_size = 25
evaluate_shots = [5, 5, 10]


# initialise model and optimiser
net_apply, net_params = create_model(rng)
opt_init, opt_update, get_params = optimizers.adam(step_size=meta_lr)
opt_state = opt_init(net_params)


def loss(params, inputs, y2):
    'compute average loss for the batch'
    predictions = net_apply(params, inputs)
    return np.mean((y2 - predictions)**2)

def inner_update(p, x1, y1):
    grads = grad(loss)(p, x1, y1)
    inner_sgd_fn = lambda g, state: (state - update_lr*g)
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


# meta-train
with tqdm(range(metatrain_iterations), total=metatrain_iterations) as pbar:
    for i in pbar:
        x1, y1, x2, y2 = meta_train_data(meta_batch_size, support_size, query_size)
        opt_state, l = step(i, opt_state, x1, y1, x2, y2)
        pbar.set_description(f'loss: {l:.5f}')

# meta-test
x2, y2 = testing_data()
net_params = get_params(opt_state)
preds = [vmap(partial(net_apply, net_params))(x2)] # zero-shot generalisation
pointsx, pointsy = [], []
for es in evaluate_shots:
    x1, y1 = training_data(es)
    pointsx.append(x1)
    pointsy.append(y1)
    net_params = inner_update(net_params, x1, y1) # train
    preds.append(vmap(partial(net_apply, net_params))(x2)) # test

# plot
plt.plot(x2, y2, label='target')
for i, p in enumerate(preds):
    plt.plot(x2, p, label=f'{sum(evaluate_shots[:i])}-shots')
plt.gca().set_prop_cycle(color=['green', 'red', 'purple'])
for px, py in zip(pointsx, pointsy):
    plt.scatter(px, py)
plt.legend()
plt.show()


# plt.plot(onp.convolve(np_maml_loss, [.05]*20), label='task_batch=1')
# plt.plot(onp.convolve(np_batched_maml_loss, [.05]*20), label='task_batch=4')
# plt.ylim(0., 1e-1)
# plt.legend()
