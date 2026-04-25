
import os
import torch
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Save directly to the models folder
SAVE_DIR = os.path.join(BASE_DIR, 'models')

Test_WGAN = os.path.join(BASE_DIR, 'Wgan Test Image Generated')
Test_DCGAN = os.path.join(BASE_DIR, 'DCGAN test images')

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 64
IMAGE_SIZE = 64
LR = 1e-4

# Checkpoints (Using PyTorch native .pth)
WGAN_CHECKPOINT = f"{SAVE_DIR}/wgan_checkpoint.pth"
DCGAN_CHECKPOINT = f"{SAVE_DIR}/dcgan_checkpoint.pth"

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_env():
    for f in [SAVE_DIR, Test_WGAN, Test_DCGAN]:
        if not os.path.exists(f):
            os.makedirs(f)
            print(f"Created folder: {f}")
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True