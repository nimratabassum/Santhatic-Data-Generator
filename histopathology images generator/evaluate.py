import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import linalg
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

from config import DEVICE, BATCH_SIZE, SAVE_DIR, WGAN_CHECKPOINT, DCGAN_CHECKPOINT, BASE_DIR
from data_loader import get_dataloaders
from models import Generator, SimpleClassifier

def calculate_fid_simple(real_images, fake_images):
    real_flat = real_images.view(real_images.size(0), -1).numpy()
    fake_flat = fake_images.view(fake_images.size(0), -1).numpy()
    mu_real, sigma_real = np.mean(real_flat, axis=0), np.cov(real_flat, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_flat, axis=0), np.cov(fake_flat, rowvar=False)
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def main():
    print("Loading data...")
    train_loader, test_loader = get_dataloaders()
    
    print("\nEvaluating FID Scores...")
    netG_wgan = Generator().to(DEVICE)
    netG_dcgan = Generator().to(DEVICE)
    
    if os.path.exists(WGAN_CHECKPOINT) and os.path.exists(DCGAN_CHECKPOINT):
        netG_wgan.load_state_dict(torch.load(WGAN_CHECKPOINT, map_location=DEVICE)['netG'])
        netG_dcgan.load_state_dict(torch.load(DCGAN_CHECKPOINT, map_location=DEVICE)['netG'])
    else:
        print("Warning: Missing checkpoints. Train both models first.")
        return

    netG_wgan.eval()
    netG_dcgan.eval()

    real_samples = []
    for data, _ in train_loader:
        real_samples.append(data)
        if len(real_samples) * BATCH_SIZE >= 1000: break
    real_samples = torch.cat(real_samples, dim=0)[:1000]

    with torch.no_grad():
        fake_wgan, fake_dcgan_samples = [], []
        for _ in range(16):  
            noise = torch.randn(BATCH_SIZE, 100, 1, 1, device=DEVICE)
            fake_wgan.append(netG_wgan(noise).cpu())
            fake_dcgan_samples.append(netG_dcgan(noise).cpu())
        fake_wgan = torch.cat(fake_wgan, dim=0)[:1000]
        fake_dcgan_samples = torch.cat(fake_dcgan_samples, dim=0)[:1000]

    fid_wgan = calculate_fid_simple(real_samples, fake_wgan)
    fid_dcgan = calculate_fid_simple(real_samples, fake_dcgan_samples)

    print("=" * 50)
    print("FID SCORES (Lower is Better)")
    print("=" * 50)
    print(f"WGAN-GP FID: {fid_wgan:.2f}")
    print(f"DCGAN FID:   {fid_dcgan:.2f}")
    print("=" * 50)

    print("\nTraining Downstream Classifier on Real Data...")
    classifier = SimpleClassifier(num_classes=9).to(DEVICE)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer_clf = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(5):
        classifier.train()
        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.squeeze().long().to(DEVICE)
            optimizer_clf.zero_grad()
            outputs = classifier(data)
            loss = criterion_ce(outputs, labels)
            loss.backward()
            optimizer_clf.step()
        print(f"Classifier Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

    classifier.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(DEVICE), labels.squeeze().long().to(DEVICE)
            outputs = classifier(data)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    print("\n" + "=" * 50)
    print("CLASSIFICATION METRICS ON TEST SET")
    print("=" * 50)
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 (Macro):  {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print("=" * 50)

    class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"{BASE_DIR}/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nEvaluation Complete! Graphs saved to '{BASE_DIR}'")

if __name__ == "__main__":
    main()