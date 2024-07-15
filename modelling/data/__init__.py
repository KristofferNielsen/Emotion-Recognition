from torch.utils.data import Dataset
from .feat_data import Data_Feat

class get_datasets(Dataset):

    def __init__(self, root, names, labels,type):
        self.dataset = Data_Feat(root, names, labels,type)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collater(self, instances):
        return self.dataset.collater(instances)
         
    def get_featdim(self):
        return self.dataset.get_featdim()