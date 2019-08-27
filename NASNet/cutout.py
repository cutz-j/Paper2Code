import numpy as np
import matplotlib.pyplot as plt

def cutout(img):
    """
    # Function: random occulusion image
    # Arguments:
        img: image
    # Returns:
        img
    """
    MAX_CUTS = 5 # chance to get more cuts
    MAX_LENGTH_MUTIPLIER = 5 # chance to get larger cuts
    # 16 for cifar10, 8 for cifar100
    
    height = img.shape[1]
    width = img.shape[2]
    
    # mean norm
    mean = img.mean(keepdims=True)
    img -= mean
    
    mask = np.ones((height, width), dtype=np.float32)
    nb_cuts = np.random.randint(0, MAX_CUTS + 1)
    
    # cutout
    for i in range(nb_cuts):
        y = np.random.randint(height)
        x = np.random.randint(width)
        length = 4 * np.random.randint(1, MAX_LENGTH_MUTIPLIER+1)
        
        y1 = np.clip(y-length//2, 0, height)
        y2 = np.clip(y+length//2, 0, height)
        x1 = np.clip(x-length//2, 0, width)
        x2 = np.clip(x+length//2, 0, width)
        
        mask[y1:y2, x1:x2] = 0.
    
    img = img * mask
    return img