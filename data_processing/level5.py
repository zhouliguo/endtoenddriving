from torch.utils.data import Dataset


class level5_dataset(Dataset):
    def __init__(self):
        self

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return