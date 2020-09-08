'''
Adapted from
https://hackernoon.com/flax-googles-open-source-approach-to-flexibility-in-machine-learning-iw9y324j
and
https://flax.readthedocs.io/en/latest/annotated_mnist.html

and
https://github.com/google/jax/issues/3382
'''


#!/usr/bin/python
# -*- coding: utf-8 -*-
import jax
import flax

import numpy as onp
import jax.numpy as jnp

import torch.utils.data
from torchvision import datasets, transforms


batch_size = 128
test_batch_size = 10000

t = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=t),
    batch_size=batch_size, shuffle=True
)
test_ds = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=t),
    batch_size=test_batch_size, shuffle=True
)


class CNN(flax.nn.Module):

    def apply(self, x):
        x = flax.nn.Conv(x, features=32, kernel_size=(3, 3))
        # x = flax.nn.relu(x)
        x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.nn.Conv(x, features=64, kernel_size=(3, 3))
        # x = flax.nn.relu(x)
        x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = flax.nn.Dense(x, features=256)
        # x = flax.nn.relu(x)
        x = flax.nn.Dense(x, features=10)
        x = flax.nn.log_softmax(x)
        return x


@jax.vmap
def cross_entropy_loss(logits, label):
    return -logits[label]


def compute_metrics(logits, labels):
    loss = jnp.mean(cross_entropy_loss(logits, labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


@jax.jit
def train_step(optimizer, batch):

    def loss_fn(model):
        logits = model(batch['image'])
        loss = jnp.mean(cross_entropy_loss(logits, batch['label']))
        return loss

    grad = jax.grad(loss_fn)(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer


@jax.jit
def eval(model, batch):
    logits = model(batch['image'])
    return compute_metrics(logits, batch['label'])


def train():

    (_, initial_params) = CNN.init_by_shape(jax.random.PRNGKey(0), [((1, 28, 28, 1), jnp.float32)])
    model = flax.nn.Model(CNN, initial_params)

    optimizer = flax.optim.Momentum(learning_rate=0.1, beta=0.9).create(model)

    for epoch in range(10):
        for batch in train_ds:
            batch = {
                'image': jnp.asarray(onp.asarray(batch[0].permute(0, 2, 3, 1))),
                'label': jnp.asarray(onp.asarray(batch[1]))
            }
            optimizer = train_step(optimizer, batch)

        batch = next(iter(test_ds))
        batch = {
            'image': jnp.asarray(onp.asarray(batch[0].permute(0, 2, 3, 1))),
            'label': jnp.asarray(onp.asarray(batch[1]))
        }
        metrics = eval(optimizer.target, batch)

        print(f"eval epoch: {epoch+1}, loss: {metrics['loss']:.4f}, "
            + f"accuracy: {100*metrics['accuracy']:.2f}")

train()

# (_, initial_params) = CNN.init_by_shape(jax.random.PRNGKey(0), [((1, 28, 28, 1), jnp.float32)])
# model = flax.nn.Model(CNN, initial_params)


# #!/usr/bin/python
# # -*- coding: utf-8 -*-
# from jax.lib import xla_bridge
# import jax
# import flax
#
# import numpy as onp
# import jax.numpy as jnp
# import csv
# # import tensorflow as tf
# # import tensorflow_datasets as tfds
#
# print(xla_bridge.get_backend().platform)
#
#
# def train():
#
#     #   train_ds = create_dataset(tf.estimator.ModeKeys.TRAIN)
#     #   test_ds = create_dataset(tf.estimator.ModeKeys.EVAL)
#     #
#     #   test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
#     #
#     # # test_ds is one giant batch
#     #
#     #   test_ds = test_ds.batch(1000)
#     #
#     # # test ds is a feature dictonary!
#     #
#     #   test_ds = tf.compat.v1.data.experimental.get_single_element(test_ds)
#     #   test_ds = tfds.as_numpy(test_ds)
#     #   test_ds = {'image': test_ds[0].astype(jnp.float32),
#     #              'label': test_ds[1].astype(jnp.int32)}
#
#     (_, initial_params) = CNN.init_by_shape(jax.random.PRNGKey(0),
#             [((1, 160, 120, 3), jnp.float32)])
#
#     model = flax.nn.Model(CNN, initial_params)
#
#     optimizer = flax.optim.Momentum(learning_rate=0.01, beta=0.9,
#                                     weight_decay=0.0005).create(model)
#
#     for epoch in range(50):
#         for batch in tfds.as_numpy(train_ds):
#             optimizer = train_step(optimizer, batch)
#
#         metrics = eval(optimizer.target, test_ds)
#
#         print('eval epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch
#                 + 1, metrics['loss'], metrics['accuracy'] * 100))
#
#
# class CNN(flax.nn.Module):
#
#     def apply(self, x):
#         x = flax.nn.Conv(x, features=128, kernel_size=(3, 3))
#         x = flax.nn.relu(x)
#         x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#         x = flax.nn.Conv(x, features=128, kernel_size=(3, 3))
#         x = flax.nn.relu(x)
#         x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#         x = flax.nn.Conv(x, features=64, kernel_size=(3, 3))
#         x = flax.nn.relu(x)
#         x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#         x = flax.nn.Conv(x, features=32, kernel_size=(3, 3))
#         x = flax.nn.relu(x)
#         x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#         x = flax.nn.Conv(x, features=16, kernel_size=(3, 3))
#         x = flax.nn.relu(x)
#         x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#         x = x.reshape((x.shape[0], -1))
#         x = flax.nn.Dense(x, features=256)
#         x = flax.nn.relu(x)
#         x = flax.nn.Dense(x, features=64)
#         x = flax.nn.relu(x)
#         x = flax.nn.Dense(x, features=4)
#         x = flax.nn.softmax(x)
#         return x
#
#
# def compute_metrics(logits, labels):
#     loss = jnp.mean(cross_entropy_loss(logits, labels))
#     accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
#     return {'loss': loss, 'accuracy': accuracy}
#
#
# @jax.vmap
# def cross_entropy_loss(logits, label):
#     return -jnp.log(logits[label])
#
#
# @jax.jit
# def train_step(optimizer, batch):
#
#     def loss_fn(model):
#         logits = model(batch[0])
#         loss = jnp.mean(cross_entropy_loss(logits, batch[1]))
#         return loss
#
#     grad = jax.grad(loss_fn)(optimizer.target)
#     optimizer = optimizer.apply_gradient(grad)
#     return optimizer
#
#
# @jax.jit
# def eval(model, eval_ds):
#     logits = model(eval_ds['image'])
#     return compute_metrics(logits, eval_ds['label'])
#
#
# train()
