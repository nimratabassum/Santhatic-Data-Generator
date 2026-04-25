import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
from config import DEVICE, DCGAN_CHECKPOINT, Test_DCGAN
from models import Generator

def generate_dcgan_images(num_images=20, save_name="dcgan_test_sample.png"):
    if not os.path.exists(DCGAN_CHECKPOINT):
        print(f"Error: Could not find model at {DCGAN_CHECKPOINT}. Train the model first.")
        return

    print("Loading trained DCGAN Generator...")
    netG = Generator().to(DEVICE)
    checkpoint = torch.load(DCGAN_CHECKPOINT, map_location=DEVICE, weights_only=False)
    netG.load_state_dict(checkpoint['netG'])
    netG.eval() 

    print(f"Generating {num_images} DCGAN images...")
    with torch.no_grad():
        noise = torch.randn(num_images, 100, 1, 1).to(DEVICE)
        fake_images = netG(noise).cpu()

    save_path = os.path.join(Test_DCGAN, save_name)
    vutils.save_image(vutils.make_grid(fake_images, padding=2, normalize=True), save_path)
    print(f"DCGAN Images saved to: {save_path}")

if __name__ == "__main__":
    generate_dcgan_images(num_images=20, save_name="dcgan_test_4.png")