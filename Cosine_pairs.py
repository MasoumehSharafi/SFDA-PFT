import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
import torch

# Feature extractor (e.g., ResNet-18)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # remove classification head
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding(image):
    with torch.no_grad():
        input_tensor = transform(image).unsqueeze(0)
        embedding = model(input_tensor).squeeze().numpy()
        return embedding / np.linalg.norm(embedding)

# Load dataset
dataset = pd.read_csv('./stressid_data/source_1fps.txt', sep=' ', names=['path', 'label'])
dataset['subject'] = dataset['path'].apply(lambda p: os.path.basename(os.path.dirname(os.path.dirname(p))))

# Extract embeddings
embeddings = {}
for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Extracting Embeddings"):
    img = cv2.imread(row['path'])
    if img is None:
        print(f"Could not load image: {row['path']}")
        continue
    try:
        emb = get_embedding(img)
        embeddings[row['path']] = emb
    except:
        print(f"Embedding failed for: {row['path']}")

# Pair images using cosine distance
with open('./stressid_data/source_cosine_pairs.txt', 'w') as f:
    for i, row1 in tqdm(dataset.iterrows(), total=len(dataset), desc="Finding Pairs"):
        path1, label1, subj1 = row1['path'], row1['label'], row1['subject']
        if path1 not in embeddings:
            continue

        max_sim = -1
        best_path = None

        for j, row2 in dataset.iterrows():
            if i == j:
                continue
            path2, label2, subj2 = row2['path'], row2['label'], row2['subject']

            if label1 == label2:
                continue  # Skip same label
            if subj1 == subj2:
                continue  # Skip same subject
            if path2 not in embeddings:
                continue

            sim = cosine_similarity(
                embeddings[path1].reshape(1, -1),
                embeddings[path2].reshape(1, -1)
            )[0][0]
            if sim > max_sim:
                max_sim = sim
                best_path = path2

        if best_path:
            f.write(f"{path1} {best_path} {max_sim:.6f}\n")
            f.flush()
