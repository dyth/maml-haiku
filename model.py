from jax.experimental import stax
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax # neural network layers

def create_model(rng):
    'returning initialization and evaluation functions'
    # then initialise network weights
    net_init, net_apply = stax.serial(
        Dense(40), Relu,
        Dense(40), Relu,
        Dense(1)
    )
    in_shape = (-1, 1,)
    out_shape, net_params = net_init(rng, in_shape)
    return net_apply, net_params
