import numpy as np
from PIL import Image 
from torch.utils.data import Dataset

class KMNIST(Dataset):
    def __init__(self, path, train):
        self.path = path
        self.images = []
        self.labels = []
        if(train):
            self.images = self.load_images(self.path + 'train-images', 60000)
            self.labels = self.load_labels(self.path + 'train-labels', 60000)
        else:
            self.images = self.load_images(self.path + 'test-images', 10000)
            self.labels = self.load_labels(self.path + 'test-labels', 10000)
    def load_images(self, file, num):
        with open(file, 'rb') as f:
            f.read(16)
            buf = f.read(28 * 28 * num)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(num, 1, 28, 28)
        return data
    def load_labels(self, file, num):
        with open(file, 'rb') as f:
            f.read(8)
            buf = f.read(num)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return data
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])
