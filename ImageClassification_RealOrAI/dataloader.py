import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

import pandas as pd
import os
from PIL import Image

class CLSDataset(Dataset):
    def __init__(self, data_dir, label_data, test=False, transform=None):
        label_data = pd.read_csv(label_data)
        
        self.files = label_data['Image'].tolist()
        if not test:
            self.label = label_data['Label'].tolist()

        self.test = test
        self.data_dir = data_dir

        if transform is not None:
            self.transform = transform
        else:
            if self.test:
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = T.Compose([
                    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(degrees=15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Output:
            - image: Transformed image tensor
            - label: Corresponding label for the image
        """
        if not self.test:
            image, label = self.files[idx], self.label[idx]
        else:
            image = self.files[idx]

        image_path = os.path.join(self.data_dir, image)
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if not self.test:
            # label = torch.LongTensor([label])
            return {'pixel_values': image, 'labels': label}
        else:
            return {'pixel_values': image}