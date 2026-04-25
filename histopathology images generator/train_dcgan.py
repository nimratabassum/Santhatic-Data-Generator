
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from config import setup_env, set_seed, DEVICE, EPOCHS, SAVE_DIR, DCGAN_CHECKPOINT, Test_DCGAN
from data_loader import get_dataloaders
from models import Generator, DCGAN_Discriminator

def main():
    setup_env()
    set_seed(42)
    train_loader, _ = get_dataloaders()
    
    netG_dcgan = Generator().to(DEVICE)
    netD_dcgan = DCGAN_Discriminator().to(DEVICE)
    
    criterion_bce = nn.BCELoss()
    optG_dcgan = optim.Adam(netG_dcgan.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optD_dcgan = optim.Adam(netD_dcgan.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    print("Training DCGAN Baseline...")
    fixed_noise = torch.randn(64, 100, 1, 1).to(DEVICE)
    dcgan_G_losses, dcgan_D_losses = [], []
    
    for epoch in range(EPOCHS):
        for i, (data, _) in enumerate(train_loader):
            real = data.to(DEVICE)
            b_size = real.size(0)
            real_label = torch.ones(b_size, 1, 1, 1, device=DEVICE)
            fake_label = torch.zeros(b_size, 1, 1, 1, device=DEVICE)
            
            # Train Discriminator
            netD_dcgan.zero_grad()
            output_real = netD_dcgan(real)
            loss_real = criterion_bce(output_real, real_label)
            
            noise = torch.randn(b_size, 100, 1, 1, device=DEVICE)
            fake = netG_dcgan(noise)
            output_fake = netD_dcgan(fake.detach())
            loss_fake = criterion_bce(output_fake, fake_label)
            
            d_loss = loss_real + loss_fake
            d_loss.backward()
            optD_dcgan.step()
            
            # Train Generator
            netG_dcgan.zero_grad()
            output = netD_dcgan(fake)
            g_loss = criterion_bce(output, real_label)
            g_loss.backward()
            optG_dcgan.step()
            
            dcgan_G_losses.append(g_loss.item())
            dcgan_D_losses.append(d_loss.item())
            
        print(f"DCGAN Epoch [{epoch+1}/{EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        
    # Save final checkpoint
    torch.save({
        'netG': netG_dcgan.state_dict(),
        'netD': netD_dcgan.state_dict(),
        'G_losses': dcgan_G_losses,
        'D_losses': dcgan_D_losses
    }, DCGAN_CHECKPOINT)
    print("DCGAN Training Complete!")
    
    # Generate and save samples
    with torch.no_grad():
        fake_dcgan = netG_dcgan(fixed_noise).detach().cpu()
        
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("DCGAN Baseline Samples (64x64)")
    plt.imshow(np.transpose(vutils.make_grid(fake_dcgan, padding=2, normalize=True, nrow=8), (1,2,0)))
    plt.savefig(f"{Test_DCGAN}/dcgan_samples.png")

if __name__ == "__main__":
    main()