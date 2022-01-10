'''
Adapted from https://flax.readthedocs.io/en/latest/annotated_mnist.html
and https://github.com/google/jax/issues/3382
'''
import jax
import flax
from jax.lib import xla_bridge
import jax.numpy as np

from jax import vmap, jit, random, grad
from tqdm.autonotebook import tqdm
import numpy as onp
from torchvision import datasets, transforms
from numpyloader import NumpyLoader, FlattenAndCast


print(xla_bridge.get_backend().platform)
rng = random.PRNGKey(0)
batch_size = 128
test_batch_size = 1000
epochs = 10


class CNN(flax.nn.Module):
    def apply(self, x):
        print(x.shape)
        x = flax.nn.Conv(x, features=32, kernel_size=(3, 3))
        x = np.maximum(x, 0)
        x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.nn.Conv(x, features=64, kernel_size=(3, 3))
        x = np.maximum(x, 0)
        x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = flax.nn.Dense(x, features=256)
        x = np.maximum(x, 0)
        x = flax.nn.Dense(x, features=10)
        x = flax.nn.log_softmax(x)
        return x


@jax.vmap
def cross_entropy_loss(logits, label):
    return -logits[label.astype(np.int32)]

def compute_metrics(logits, labels):
    loss = np.mean(cross_entropy_loss(logits, labels))
    accuracy = np.mean(np.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

@jax.jit
def eval(model, batch):
    logits = model(batch[0])
    return compute_metrics(logits, batch[1])

@jax.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        logits = model(batch[0])
        loss = np.mean(cross_entropy_loss(logits, batch[1]))
        return loss
    grad = jax.grad(loss_fn)(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, eval(optimizer.target, batch)


t = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    FlattenAndCast()
])

train_data = datasets.MNIST('data', train=True, download=True, transform=t)
test_data = datasets.MNIST('data', train=False, download=True, transform=t)
train_ds = NumpyLoader(train_data, batch_size=batch_size, shuffle=True)
test_ds = NumpyLoader(test_data, batch_size=test_batch_size, shuffle=True)

(_, initial_params) = CNN.init_by_shape(rng, [((1, 28, 28, 1), np.float32)])
model = flax.nn.Model(CNN, initial_params)
optimizer = flax.optim.Momentum(learning_rate=0.1, beta=0.9).create(model)

for epoch in range(epochs):
    print(epoch)

    # training
    with tqdm(train_ds, total=len(train_ds)) as pbar:
        for batch in pbar:
            optimizer, metrics = train_step(optimizer, batch)
            pbar.set_description(f"loss: {metrics['loss']:.4f}, accuracy: {100*metrics['accuracy']:.2f}")

    # testing
    with tqdm(test_ds, total=len(test_ds)) as pbar:
        for batch in pbar:
            metrics = eval(optimizer.target, batch)
            pbar.set_description(f"loss: {metrics['loss']:.4f}, accuracy: {100*metrics['accuracy']:.2f}")
