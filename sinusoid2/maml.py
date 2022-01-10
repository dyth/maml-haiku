import argparse, time, pickle
import random as orandom
import numpy as onp
import haiku as hk
import jax
import optax
import subprocess

from functools import partial
from distutils.util import strtobool
from jax import numpy as np
from jax import nn, random, ops
from jax.lib import xla_bridge
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten
from haiku import data_structures as ds
from jax import vmap
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt

from sin_data import meta_train_data, training_data, testing_data



# hyperparameters from https://github.com/cbfinn/maml/blob/master/main.py
strtobool = lambda x: bool(strtobool(x))
parser    = argparse.ArgumentParser()
parser.add_argument('--name',          type=str,            default='maml-haiku')
parser.add_argument('--outer-lr',      type=float,          default=1e-3)
parser.add_argument('--inner-lr',      type=float,          default=1e-2)
parser.add_argument('--outer-steps',   type=int,            default=70000)
parser.add_argument('--outer-batch',   type=int,            default=25)
parser.add_argument('--support-batch', type=int,            default=5)
parser.add_argument('--query-batch',   type=int,            default=5)
parser.add_argument('--eval-shots',    type=int, nargs='+', default=[5, 5, 10])
# parser.add_argument('--something',     type=strtobool, default=True)
args = parser.parse_args()


print('Backend:', xla_bridge.get_backend().platform)
seed = random.PRNGKey(0)



class Model(hk.Module):
    def __call__(self, x):
        x = hk.Linear(40)(x)
        x = nn.relu(x)
        x = hk.Linear(40)(x)
        x = nn.relu(x)
        x = hk.Linear(1)(x)
        return x

def model(x):
    return Model()(x)


model           = hk.transform_with_state(model)
init_seed, seed = random.split(seed)
params, state   = model.init(init_seed, random.normal(init_seed, (1,)))
opt             = optax.adam(args.outer_lr)
opt_state       = opt.init(params)
apply           = jax.jit(model.apply)
# apply           = model.apply



def mse_loss(params, state, x, y):
    preds, state = apply(params, state, None, x)
    mse          = np.mean((y - preds)**2)
    return mse, state

def inner_update(params, state, x, y):
    grads, state = jax.grad(mse_loss, has_aux=True)(params, state, x, y)
    inner_sgd    = lambda p, g: (p - args.inner_lr*g)
    new_params   = jax.tree_util.tree_multimap(inner_sgd, params, grads)
    return new_params, state

def outer_loss(params, state, support, query):
    new_params, state = inner_update(params, state, *support)
    query_loss, _     = mse_loss(new_params, state, *query)
    return query_loss, state

def batch_outer_loss(params, state, support, query):
    task_loss, state = vmap(partial(outer_loss, params, state))(support, query)
    return task_loss.mean(), state

@jax.jit
def outer_update(params, state, opt_state, support, query):
    (loss, state), grads = jax.value_and_grad(batch_outer_loss, has_aux=True)(params, state, support, query)
    grads, opt_state     = opt.update(grads, opt_state)
    params               = optax.apply_updates(params, grads)
    return loss, params, state, opt_state


for _ in range(args.outer_steps):
    data = meta_train_data(args.outer_batch, args.support_batch, args.query_batch)
    loss, params, state, opt_state = outer_update(params, state, opt_state, *data)


# meta-test
x_test, y_test = testing_data()
preds          = [apply(params, state, None, x_test)[0]]
xs, ys         = [], []
for es in args.eval_shots:
    x, y = training_data(es)
    xs.append(x)
    ys.append(y)
    params, state = inner_update(params, state, x, y) # train
    preds.append(apply(params, state, None, x_test)[0]) # test


# plot
plt.plot(x_test, y_test, label='target')
for i, pred in enumerate(preds):
    plt.plot(x_test, pred, label=f'{sum(args.eval_shots[:i])}-shot')
plt.gca().set_prop_cycle(color=['green', 'red', 'purple'])
for px, py in zip(xs, ys):
    plt.scatter(px, py)
plt.legend()
plt.show()
