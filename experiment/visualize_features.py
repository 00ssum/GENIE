# visualize_features.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import os

def plot_2d(data, labels, title, path):
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(data[idx, 0], data[idx, 1], label=f'Class {label}', alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

path= os.path.join("/jsm0707/GENIE/train_output","PACS","GENIE","[3]","250330_00-59-24_resnet50_sgd")

# Load features
before = np.load(path+"/features_before.npy")
after = np.load(path+"/features_after.npy")
labels = np.load(path+"/labels_before.npy")  # 동일

# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
before_umap = umap_model.fit_transform(before)
after_umap = umap_model.fit_transform(after)

plot_2d(before_umap, labels, "UMAP - Before Training", path+"/umap_before.png")
plot_2d(after_umap, labels, "UMAP - After Training", path+"/umap_after.png")

# t-SNE
tsne_model = TSNE(n_components=2, random_state=42)
before_tsne = tsne_model.fit_transform(before)
after_tsne = tsne_model.fit_transform(after)

plot_2d(before_tsne, labels, "t-SNE - Before Training", path+"/tsne_before.png")
plot_2d(after_tsne, labels, "t-SNE - After Training", path+"/tsne_after.png")
