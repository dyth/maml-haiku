import torchmeta
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import MetaDataLoader

import jax
import flax
from jax.lib import xla_bridge
from jax import vmap, jit, random, grad
from jax import numpy as np

import numpy as onp
from tqdm.autonotebook import tqdm
from torchvision import datasets, transforms
from numpyloader import numpy_collate#NumpyLoader, FlattenAndCast


print(xla_bridge.get_backend().platform)
rng = random.PRNGKey(0)
epochs = 10
support_size = 64
# batch_size = 128
# test_batch_size = 1000






training_dataset = omniglot("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
testing_dataset = omniglot("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
training_dataloader = MetaDataLoader(training_dataset, batch_size=support_size, collate_fn=numpy_collate, num_workers=4)
testing_dataloader = MetaDataLoader(testing_dataset, batch_size=support_size, collate_fn=numpy_collate, num_workers=4)



for batch in training_dataloader:
	train_inputs, train_targets = batch["train"]
	print('Train inputs shape: {0}'.format(train_inputs.shape))
	print('Train targets shape: {0}'.format(train_targets.shape))

	test_inputs, test_targets = batch["test"]
	print('Test inputs shape: {0}'.format(test_inputs.shape))
	print('Test targets shape: {0}'.format(test_targets.shape))
	print(c)

# python train.py /path/to/data --dataset omniglot --num-ways 5 --num-shots 1
# --step-size 0.4 --batch-size 32 --num-epochs 600
