import torch.nn as nn
from torch import flatten

class LeNet(nn.Module):
	def __init__(self, numChannels, classes):
		super().__init__()
		self.layers1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
			nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
		self.flatten = nn.Flatten()
		self.layers2 = nn.Sequential(
			nn.Linear(in_features=800, out_features=500),
			nn.ReLU(),
			nn.Linear(in_features=500, out_features=10),
			nn.LogSoftmax(dim=1)
        )
	def forward(self, x):
		out = self.layers1(x)
		out = self.flatten(out)
		out = self.layers2(out)
		# return the output predictions
		return out