from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, x, s):
        self.x = x
        if s is None:
            self.s = [None] * len(x)
        else:
            self.s = s
        assert len(x) == len(self.s)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.s[i]  # [1:]


class Dataset:
    def __init__(self, data, inputs):
        self.data = data
        self.inputs = inputs

    def get_dataloader(self):
        return DataLoader(Data(self.data, self.inputs), batch_size=None)
