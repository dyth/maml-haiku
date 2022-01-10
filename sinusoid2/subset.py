import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np



num_classes = 10

def f(x):
  return hk.nets.MLP([300, 100, num_classes])(x)

f = hk.transform(f)

def test(params, num_classes=num_classes):
  x = np.arange(num_classes).reshape([num_classes, 1]).astype(np.float32)
  y = jnp.argmax(f.apply(params, None, x), axis=-1)
  for x, y in zip(x, y):
    print(x, "->", y)

rng = jax.random.PRNGKey(42)
x = np.zeros([num_classes, 1])
params = f.init(rng, x)

print("before training")
test(params)



def dataset(*, batch_size, num_records):
  for _ in range(num_records):
    y = np.arange(num_classes)
    y = np.random.permutation(y)[:batch_size]
    x = y.reshape([batch_size, 1]).astype(np.float32)
    yield x, y

for x, y in dataset(batch_size=4, num_records=5):
  print("x :=", x.tolist(), "y :=", y)



# Partition our params into trainable and non trainable explicitly.
trainable_params, non_trainable_params = hk.data_structures.partition(
    lambda m, n, p: m != "mlp/~/linear_1", params)

print("trainable:", list(trainable_params))
print("non_trainable:", list(non_trainable_params))



def loss_fn(trainable_params, non_trainable_params, images, labels):
  # NOTE: We need to combine trainable and non trainable before calling apply.
  params = hk.data_structures.merge(trainable_params, non_trainable_params)

  # NOTE: From here on this is a standard softmax cross entropy loss.
  logits = f.apply(params, None, images)
  labels = jax.nn.one_hot(labels, logits.shape[-1])
  return -jnp.sum(labels * jax.nn.log_softmax(logits)) / labels.shape[0]

def sgd_step(params, grads, *, lr):
  return jax.tree_multimap(lambda p, g: p - g * lr, params, grads)

def train_step(trainable_params, non_trainable_params, x, y):
  # NOTE: We will only compute gradients wrt `trainable_params`.
  trainable_params_grads = jax.grad(loss_fn)(trainable_params,
                                             non_trainable_params, x, y)

  # NOTE: We are only updating `trainable_params`.
  print(type(trainable_params), type(trainable_params_grads))
  trainable_params = sgd_step(trainable_params, trainable_params_grads, lr=0.1)
  return trainable_params

train_step = jax.jit(train_step)

for x, y in dataset(batch_size=num_classes, num_records=10000):
  # NOTE: In our training loop only our trainable parameters are updated.
  trainable_params = train_step(trainable_params, non_trainable_params, x, y)



# Merge params again for inference.
params = hk.data_structures.merge(trainable_params, non_trainable_params)

print("after training")
test(params)


test(params, num_classes=num_classes+10)
