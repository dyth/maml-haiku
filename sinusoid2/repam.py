'''
Adapted from https://jax.readthedocs.io/en/stable/notebooks/maml.html
'''
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

model           = hk.transform(model)
init_seed, seed = random.split(seed)
random_input    = random.normal(init_seed, (1,))
outer_params    = model.init(init_seed, random_input)

outer_params, op_unravel = ravel_pytree(outer_params)
outer_opt                = optax.adam(args.outer_lr)
outer_opt_state          = outer_opt.init(outer_params)

# @jax.jit
def outer_apply(outer_params, x):
    outer_params = op_unravel(outer_params)
    y       = model.apply(outer_params, None, x)
    # flat_op = ravel_pytree(outer_params)[0]
    # y       = model.apply(flat_op, None, x)
    return y



class Reparameterize(hk.Module):
    def __call__(self, flat_params):
        fp_shape = flat_params.shape
        stdev    = np.sqrt(fp_shape[0])
        init     = hk.initializers.TruncatedNormal(1. / stdev)
        params_w = hk.get_parameter('w', shape=fp_shape, init=init)
        params_b = hk.get_parameter('b', shape=fp_shape, init=init)
        reparams = params_w * flat_params  +  params_b
        return reparams

def reparameterize(x):
    return Reparameterize()(x)

reparameterize  = hk.transform(reparameterize)
# inner_apply     = jax.jit(reparameterize.apply)
inner_apply     = reparameterize.apply
init_seed, seed = random.split(seed)
random_input    = random.normal(init_seed, outer_params.shape)
inner_params    = reparameterize.init(init_seed, random_input)
# inner_apply(inner_params, None, outer_params)

inner_opt       = optax.adam(args.inner_lr)
inner_opt_state = inner_opt.init(inner_params)


#####################################################################3

def inner_loss(inner_params, outer_params, x, y):
    # print('inner', jax.tree_map(lambda x: x.shape, inner_params))
    # print('outer', jax.tree_map(lambda x: x.shape, outer_params))
    reparams = inner_apply(inner_params, None, outer_params)
    # print('reparams', jax.tree_map(lambda x: x.shape, reparams))
    preds    = outer_apply(reparams, x)
    # print('preds', jax.tree_map(lambda x: x.shape, preds))
    mse      = np.mean((y - preds)**2)
    return mse

def inner_update(inner_params, inner_opt_state, outer_params, batch):
    # loss, grads            = jax.value_and_grad(inner_loss)(inner_params, outer_params, *batch)
    # grads, inner_opt_state = inner_opt.update(grads, inner_opt_state)
    # inner_params           = optax.apply_updates(inner_params, grads)
    # return inner_params
    loss, grads  = jax.value_and_grad(inner_loss)(inner_params, outer_params, *batch)
    inner_sgd    = lambda p, g: (p - args.inner_lr*g)
    inner_params = jax.tree_util.tree_multimap(inner_sgd, params, grads)
    return inner_params

def outer_loss(outer_params, inner_params, inner_opt_state, support, query):
    # inner_params = inner_update(inner_params, inner_opt_state, outer_params, support)
    query_loss   = inner_loss(inner_params, outer_params, *query)
    return query_loss

def batch_outer_loss(outer_params, inner_params, inner_opt_state, support, query):
    task_loss = vmap(partial(outer_loss, outer_params, inner_params, inner_opt_state))(support, query)
    # print('task_loss', task_loss.shape)
    return task_loss.mean()

# @jax.jit
def outer_update(outer_params, outer_opt_state, inner_params, inner_opt_state, support, query):
    loss, grads            = jax.value_and_grad(batch_outer_loss)(outer_params, inner_params, inner_opt_state, support, query)
    grads, outer_opt_state = outer_opt.update(grads, outer_opt_state)
    outer_params           = optax.apply_updates(outer_params, grads)
    return loss, outer_params, outer_opt_state


for _ in range(args.outer_steps):
    data = meta_train_data(args.outer_batch, args.support_batch, args.query_batch)
    # print(outer_params)
    loss, outer_params, outer_opt_state = outer_update(outer_params, outer_opt_state, inner_params, inner_opt_state, *data)
    print(loss)
    print(outer_params)
    print(c)


# meta-test
x_test, y_test = testing_data()
reparams       = inner_apply(inner_params, None, outer_params)
preds          = [outer_apply(reparams, x_test)]
xs, ys         = [], []
for es in args.eval_shots:
    x, y = training_data(es)
    xs.append(x)
    ys.append(y)
    # train
    inner_params = inner_update(inner_params, inner_opt_state, outer_params, (x, y))
    # test
    reparams = inner_apply(inner_params, None, outer_params)
    preds.append(outer_apply(reparams, x_test))
    # params = inner_update(params, x, y) # train
    # preds.append(apply(params, None, x_test)[0]) # test


# plot
plt.plot(x_test, y_test, label='target')
for i, pred in enumerate(preds):
    plt.plot(x_test, pred, label=f'{sum(args.eval_shots[:i])}-shot')
plt.gca().set_prop_cycle(color=['green', 'red', 'purple'])
for px, py in zip(xs, ys):
    plt.scatter(px, py)
plt.legend()
plt.show()
