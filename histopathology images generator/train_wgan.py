
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from config import setup_env, set_seed, DEVICE, EPOCHS, LR, WGAN_CHECKPOINT, SAVE_DIR, Test_WGAN
from data_loader import get_dataloaders
from models import Generator, Critic

def compute_gradient_penalty(netC, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1).to(DEVICE)
    interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
    d_interpolates = netC(interpolates)
    fake_grad = torch.full((real.size(0), 1, 1, 1), 1.0, requires_grad=False, device=DEVICE)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates, grad_outputs=fake_grad,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

def main():
    setup_env()
    set_seed()
    
    train_loader, _ = get_dataloaders()
    
    netG = Generator().to(DEVICE)
    netC = Critic().to(DEVICE)
    optG = optim.Adam(netG.parameters(), lr=LR, betas=(0.0, 0.9))
    optC = optim.Adam(netC.parameters(), lr=LR, betas=(0.0, 0.9))
    
    start_epoch = 0
    G_losses, C_losses = [], []
    
    if os.path.exists(WGAN_CHECKPOINT):
        print("--> Found WGAN checkpoint! Resuming...")
        checkpoint = torch.load(WGAN_CHECKPOINT, map_location=DEVICE)
        netG.load_state_dict(checkpoint['netG'])
        netC.load_state_dict(checkpoint['netC'])
        optG.load_state_dict(checkpoint['optG'])
        optC.load_state_dict(checkpoint['optC'])
        start_epoch = checkpoint['epoch']
        G_losses = checkpoint.get('G_losses', [])
        C_losses = checkpoint.get('C_losses', [])
    
    print(f"Starting WGAN Training (Epoch {start_epoch} to {EPOCHS})...")
    
    for epoch in range(start_epoch, EPOCHS):
        for i, (data, _) in enumerate(train_loader):
            real = data.to(DEVICE)
            
            # Train Critic
            for _ in range(5):
                netC.zero_grad()
                noise = torch.randn(real.size(0), 100, 1, 1).to(DEVICE)
                fake = netG(noise).detach()
                gp = compute_gradient_penalty(netC, real, fake)
                d_loss = -torch.mean(netC(real)) + torch.mean(netC(fake)) + gp
                d_loss.backward()
                optC.step()
                
            # Train Generator
            netG.zero_grad()
            noise = torch.randn(real.size(0), 100, 1, 1).to(DEVICE)
            g_loss = -torch.mean(netC(netG(noise)))
            g_loss.backward()
            optG.step()
            
            G_losses.append(g_loss.item())
            C_losses.append(d_loss.item())
            
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {i} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
                
        # Save Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'netG': netG.state_dict(),
            'netC': netC.state_dict(),
            'optG': optG.state_dict(),
            'optC': optC.state_dict(),
            'G_losses': G_losses,
            'C_losses': C_losses
        }, WGAN_CHECKPOINT)
        
    print("Training Complete. Saving loss graph...")
    plt.figure(figsize=(10,5))
    plt.plot(G_losses, label="G Loss")
    plt.plot(C_losses, label="C Loss")
    plt.legend()
    plt.savefig(f"{Test_WGAN}/wgan_loss_graph.png")
    
if __name__ == "__main__":
    main()