import os
import numpy as np
from datasets import load_dataset

def download_imagenet_subset(num_images=64, output_dir="imagenet_samples"):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Streaming {num_images} images from Hugging Face (timm/mini-imagenet)...")
    
    # "streaming=True" allows us to load data without downloading the whole dataset
    dataset = load_dataset("timm/mini-imagenet", split="train", streaming=True)
    
    count = 0
    # Iterate through the stream
    imgs = []
    for sample in dataset:
        if count >= num_images:
            break
            
        image = sample['image']
        label = sample['label']
        imgs.append(image.resize((256,256)))
        count += 1
    imgs = np.stack([np.array(img) for img in imgs])/255.
    return imgs

