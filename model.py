from jax.experimental import stax
from jax.experimental.stax import Dense, Relu


def create_model(rng):
    # create initialization and evaluation functions
    net_init, net_apply = stax.serial(
        Dense(40), Relu,
        Dense(40), Relu,
        Dense(1)
    )
    # initialise network weights
    in_shape = (-1, 1,)
    out_shape, net_params = net_init(rng, in_shape)
    return net_apply, net_params
