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
from jax.tree_util import tree_multimap  # Element-wise manipulation of collections of numpy arrays

from model import create_model

rng = random.PRNGKey(0)
alpha = .1
support_size = 20
testing_size = 20
meta_training_epochs = 20000


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

@jit
def step(i, opt_state, x1, y1, x2, y2):
    p = get_params(opt_state)
    g = grad(maml_loss)(p, x1, y1, x2, y2)
    l = maml_loss(p, x1, y1, x2, y2)
    return opt_update(i, g, opt_state), l


# initialise model and optimiser
# this LR seems to be better than 1e-2 and 1e-4
net_apply, net_params = create_model(rng)
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
opt_state = opt_init(net_params)

# train with Adam
np_maml_loss = []
for i in range(meta_training_epochs):
    # randomly sample task data
    A = onp.random.uniform(low=0.1, high=.5)
    phase = onp.random.uniform(low=0., high=np.pi)
    # generate support set data of size support_size
    x1 = onp.random.uniform(low=-5., high=5., size=(support_size, 1))
    y1 = A * onp.sin(x1 + phase)
    # generate query set data of size 1
    x2 = onp.random.uniform(low=-5., high=5.)
    y2 = A * onp.sin(x2 + phase)
    # train
    opt_state, l = step(i, opt_state, x1, y1, x2, y2)
    np_maml_loss.append(l)
    if i % 1000 == 0:
        print(i)

# generate pre-meta-testing data to see what weights look like
net_params = get_params(opt_state)
xrange_inputs = np.linspace(-5, 5, 100).reshape((100, 1))
targets = np.sin(xrange_inputs)
preds_before_meta_testing = vmap(partial(net_apply, net_params))(xrange_inputs)

# meta-testing time
x1 = onp.random.uniform(low=-5., high=5., size=(testing_size, 1))
y1 = 1. * onp.sin(x1 + 0.)
preds = []
for i in range(1, 5):
    net_params = inner_update(net_params, x1, y1)
    preds.append(vmap(partial(net_apply, net_params))(xrange_inputs))

# plot
plt.plot(xrange_inputs, preds_before_meta_testing, label='pre-meta-testing')
plt.plot(xrange_inputs, targets, label='target')
for i, preds in enumerate(preds):
    plt.plot(xrange_inputs, preds, label='{}-shot'.format(i))
plt.legend()
plt.show()
