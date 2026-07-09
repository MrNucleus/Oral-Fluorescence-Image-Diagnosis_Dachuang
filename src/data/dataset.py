import os
from PIL import Image
from torch.utils.data import Dataset


class OralFluorescenceDataset(Dataset):
    def __init__(self, data_root, split_file, transform=None, class_names=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        self.labels = []
        if class_names is None:
            class_names = ["Normal", "OLK", "OLP", "OSCC", "Other"]
        self.class_to_idx = {n: i for i, n in enumerate(class_names)}
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1] in self.class_to_idx:
                    self.samples.append(parts[0])
                    self.labels.append(self.class_to_idx[parts[1]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.samples[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
