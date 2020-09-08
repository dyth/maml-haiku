import torchmeta
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

import numpy as onp
from jax import numpy as np


support_size = 16

training_dataset = omniglot("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
testing_dataset = omniglot("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
training_dataloader = BatchMetaDataLoader(training_dataset, batch_size=support_size, num_workers=4)
testing_dataloader = BatchMetaDataLoader(testing_dataset, batch_size=support_size, num_workers=4)



for batch in training_dataloader:
	train_inputs, train_targets = batch["train"]
	print(type(np.asarray(onp.asarray(train_inputs))), type(train_targets))
	print('Train inputs shape: {0}'.format(train_inputs.shape))
	print('Train targets shape: {0}'.format(train_targets.shape))

	test_inputs, test_targets = batch["test"]
	print(type(test_inputs), type(test_targets))
	print('Test inputs shape: {0}'.format(test_inputs.shape))
	print('Test targets shape: {0}'.format(test_targets.shape))
	print(c)

# python train.py /path/to/data --dataset omniglot --num-ways 5 --num-shots 1
# --step-size 0.4 --batch-size 32 --num-epochs 600
