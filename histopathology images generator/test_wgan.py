import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
from config import DEVICE, WGAN_CHECKPOINT, Test_WGAN
from models import Generator

def generate_wgan_images(num_images=10, save_name="wgan_test_sample.png"):
    if not os.path.exists(WGAN_CHECKPOINT):
        print(f"Error: Could not find model at {WGAN_CHECKPOINT}. Train the model first.")
        return

    print("Loading trained WGAN-GP Generator...")
    netG = Generator().to(DEVICE)
    checkpoint = torch.load(WGAN_CHECKPOINT, map_location=DEVICE, weights_only=False)
    netG.load_state_dict(checkpoint['netG'])
    netG.eval() 

    print(f"Generating {num_images} WGAN images...")
    with torch.no_grad():
        noise = torch.randn(num_images, 100, 1, 1).to(DEVICE)
        fake_images = netG(noise).cpu()

    save_path = os.path.join(Test_WGAN, save_name)
    vutils.save_image(vutils.make_grid(fake_images, padding=2, normalize=True), save_path)
    print(f"WGAN Images saved to: {save_path}")

if __name__ == "__main__":
    generate_wgan_images(num_images=10, save_name="wgan_test_4.png")