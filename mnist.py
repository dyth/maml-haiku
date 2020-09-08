'''
Adapted from https://flax.readthedocs.io/en/latest/annotated_mnist.html

and
https://github.com/google/jax/issues/3382
'''
import jax
import flax
from jax.lib import xla_bridge
import jax.numpy as np

from jax import vmap, jit, random, grad
from tqdm.autonotebook import tqdm
import numpy as onp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


print(xla_bridge.get_backend().platform)
rng = random.PRNGKey(0)
batch_size = 128
test_batch_size = 1000
epochs = 10


class CNN(flax.nn.Module):
    def apply(self, x):
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
    return -logits[label]

def compute_metrics(logits, labels):
    loss = np.mean(cross_entropy_loss(logits, labels))
    accuracy = np.mean(np.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

@jax.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        # logits = model(batch['image'])
        logits = model(batch[0])
        # loss = np.mean(cross_entropy_loss(logits, batch['label']))
        loss = np.mean(cross_entropy_loss(logits, batch[1]))
        return loss
    grad = jax.grad(loss_fn)(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer

@jax.jit
def eval(model, batch):
    # logits = model(batch['image'])
    logits = model(batch[0])
    # return compute_metrics(logits, batch['label'])
    return compute_metrics(logits, batch[1])


#
# class ToJax(object):
#     def __call__(self, batch):
#         print(batch.shape)
#         return np.asarray(onp.asarray(batch.permute(0, 2, 3, 1)))
#
#     def __repr__(self):
#         return self.__class__.__name__ + '()'

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        pic = pic.permute(1, 2, 0)
        # return onp.ravel(onp.array(pic, dtype=np.float32))
        return np.array(pic, dtype=np.float32)


t = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    FlattenAndCast()
    # ToJax()
])

train_data = datasets.MNIST('data', train=True, download=True, transform=t)
test_data = datasets.MNIST('data', train=False, download=True, transform=t)
train_ds = NumpyLoader(train_data, batch_size=batch_size, shuffle=True)
test_ds = NumpyLoader(test_data, batch_size=test_batch_size, shuffle=True)


(_, initial_params) = CNN.init_by_shape(rng, [((1, 28, 28, 1), np.float32)])
model = flax.nn.Model(CNN, initial_params)
optimizer = flax.optim.Momentum(learning_rate=0.1, beta=0.9).create(model)

for epoch in range(epochs):
    for batch in train_ds:
        # batch = {
        #     'image': np.asarray(onp.asarray(batch[0].permute(0, 2, 3, 1))),
        #     'label': np.asarray(onp.asarray(batch[1]))
        # }
        optimizer = train_step(optimizer, batch)

    batch = next(iter(test_ds))
    # batch = {
    #     'image': np.asarray(onp.asarray(batch[0].permute(0, 2, 3, 1))),
    #     'label': np.asarray(onp.asarray(batch[1]))
    # }
    metrics = eval(optimizer.target, batch)

    print(f"eval epoch: {epoch+1}, loss: {metrics['loss']:.4f}, "
        + f"accuracy: {100*metrics['accuracy']:.2f}")
