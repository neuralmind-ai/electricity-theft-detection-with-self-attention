import numpy as np
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.xnan = x.copy()
        self.y = y
        self.x[np.isnan(self.x)] = 0.
        self.xnan[~np.isnan(self.xnan)] = 0.
        self.xnan[np.isnan(self.xnan)]  = 1.
        self.x = np.stack((self.x, self.xnan), axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x.astype(np.float32), y.astype(np.int64)


class CNNDataset(FraudDataset):

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        x = x.reshape(2,-1,7)
        return x.astype(np.float32), y.astype(np.int64)