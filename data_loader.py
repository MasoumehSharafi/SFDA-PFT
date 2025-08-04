# --- src/data_loader.py ---
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PairDataset(Dataset):
    def __init__(self, folder1, folder2):
        self.folder1 = folder1
        self.folder2 = folder2

        self.files1 = sorted(os.listdir(folder1))
        self.files2 = sorted(os.listdir(folder2))

        assert len(self.files1) == len(self.files2), \
            f"Mismatch: {len(self.files1)} files in folder1 vs {len(self.files2)} files in folder2"

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files1)

    def __getitem__(self, idx):
        fname1 = self.files1[idx]
        fname2 = self.files2[idx]

        img1_path = os.path.join(self.folder1, fname1)
        img2_path = os.path.join(self.folder2, fname2)

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        # label based on first folder's file name (as you did before)
        label = 0 if '_N' in fname1 else 1  

        return img1, img2, label, fname1  # keeping fname1 as the identifier

class TargetDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.filenames = sorted(os.listdir(folder))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = self.transform(Image.open(os.path.join(self.folder, fname)).convert('RGB'))
        label = 0 if '_N' in fname else 1
        return img, label, fname