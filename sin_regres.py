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

from model import create_model

rng = random.PRNGKey(0)
batch_size = 100
epochs = 100


def loss(params, inputs, targets):
    'compute average loss for the batch'
    predictions = net_apply(params, inputs)
    return np.mean((targets - predictions)**2)

@jit
def step(i, opt_state, x1, y1):
    'define a compiled update step'
    p = get_params(opt_state)
    g = grad(loss)(p, x1, y1)
    return opt_update(i, g, opt_state)

# initialise model and optimiser
net_apply, net_params = create_model(rng)
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
opt_state = opt_init(net_params)

# create data and pass through initialised network
xrange_inputs = np.linspace(-5, 5, batch_size).reshape((batch_size, 1))
targets = np.sin(xrange_inputs)
preds_before = vmap(partial(net_apply, net_params))(xrange_inputs)

# train for epochs and pass data through trained weights
for i in range(epochs):
    opt_state = step(i, opt_state, xrange_inputs, targets)
net_params = get_params(opt_state)
preds_after = vmap(partial(net_apply, net_params))(xrange_inputs)

# plot all the data
plt.plot(xrange_inputs, targets, label='target')
plt.plot(xrange_inputs, preds_before, label='prediction_before')
plt.plot(xrange_inputs, preds_after, label='prediction_after')
plt.legend()
plt.show()
