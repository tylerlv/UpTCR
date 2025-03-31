from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

def split_dataset_by_epitope(dataset, test_size=0.2):
    grouped = dataset.data.groupby('Epitope')
    
    train_indices = []
    val_indices = []

    for epitope, group in grouped:
        indices = group.index.tolist()
        if len(indices) > 1:
            train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
        else:
            train_indices.extend(indices)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset