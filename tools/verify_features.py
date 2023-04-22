import os
import torch
import numpy as np
import cv2

feature_root_path = "../data/feature/glue"
task_name ="sst"
evaluate = False

img_features = torch.load(os.path.join(feature_root_path, task_name + ("_dev" if evaluate else "_train") + ".pt"))[0] # (# samples, feature dim 255)
txt_features = torch.load(os.path.join(feature_root_path, task_name + ("_textual_dev" if evaluate else "_textual_train") + ".pt"))[0] # (# samples, feature dim 255)

print(f"img features: type {type(img_features)} | shape {img_features.size()} | range ({img_features.min()}, {img_features.max()})")
print(f"txt features: type {type(txt_features)} | shape {txt_features.size()} | range ({txt_features.min()}, {txt_features.max()})")

# similarity scores between image features and text features
scores = (img_features @ txt_features.T).numpy() # (# img samples, # text samples)
# scale/shift scores to [0, 1]
scores = (scores - scores.min()) / (scores.max() - scores.min())

# bright = similar
cv2.imwrite(os.path.join(feature_root_path, "scores" + ("_dev" if evaluate else "_train") + ".png"), (scores*255).astype(np.uint8))
