# Histopathology Image Generation using GANs

Histopathology Image Generation is a deep learning-based tool designed to automatically generate synthetic medical histopathology images. It implements and compares two Generative Adversarial Network (GAN) architectures: **DCGAN** (Deep Convolutional GAN) and **WGAN-GP** (Wasserstein GAN with Gradient Penalty).

The models are trained using the **PathMNIST** dataset (from the MedMNIST collection), which consists of colon pathology images categorized into 9 distinct tissue types.

The tool offers:
- Speedy synthetic image generation after the first download of pretrained model weights.
- Ease of use through a simple command-line interface — just copy and paste the commands, and run.

It includes:
- Fully modular source code for global configuration, data loading, and PyTorch architectures.
- Training scripts for both DCGAN and WGAN-GP with automatic checkpointing.
- Inference (test) scripts to quickly generate sample grids from trained models.

## Installation & Setup

This project has been tested with Python 3.10. See `requirements.txt` for the full list of dependencies.

To install the application, you can just clone this repository and use pip.

**Clone this repository**
```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

**(Optional) Create a virtual environment**
```bash
conda create -n gan_env python=3.10
conda activate gan_env
```

**Install dependencies**
Install the required packages using pip.
```bash
pip install -r requirements.txt
```

## Usage

All Python scripts should be run from the root directory of the project. The configurations automatically resolve absolute paths to save outputs correctly. The dataset will be automatically downloaded upon the first run.

### Training the Models

**To train WGAN-GP:**
```bash
python "histopathology image generation/train_wgan.py"
```
*(Checkpoints will be saved as `models/wgan_checkpoint.pth` and loss graphs will be saved to the WGAN test folder.)*

**To train DCGAN:**
```bash
python "histopathology image generation/train_dcgan.py"
```
*(Checkpoints will be saved as `models/dcgan_checkpoint.pth` and sample grids will be saved to the DCGAN test folder.)*

### Generating Synthetic Images (Inference)

Once the models are trained (or if you already have the `.pth` files in the `models/` directory), you can generate new synthetic medical images.

**Generate from WGAN-GP:**
```bash
python "histopathology image generation/test_wgan.py"
```
*(Generated images will be exported to `Wgan Test Image Generated/`)*

**Generate from DCGAN:**
```bash
python "histopathology image generation/test_dcgan.py"
```
*(Generated images will be exported to `DCGAN test images Generated/`)*

## Example Output

Generated images are saved to directly to your configured output folders:
- `DCGAN test images Generated/`
- `Wgan Test Image Generated/`

 *(You can add an actual image link here, e.g., `![WGAN-GP Samples](Wgan Test Image Generated/sample.png)`)*

## Model Architectures and Documentation

The repository is organized into a clean, modular structure. Below is an overview of the architecture implementations:

- **Generator (Shared):** A Deep Convolutional network utilizing `ConvTranspose2d`, `BatchNorm2d`, and `ReLU` activations, mapping a 100-dimensional latent noise vector into a 3x64x64 synthetic image using a `Tanh` output layer.
- **DCGAN Discriminator:** Standard binary classifier distinguishing between real and fake images using `BCELoss` and `Sigmoid`.
- **WGAN Critic:** Evaluates the Earth Mover's (Wasserstein) distance. Uses `InstanceNorm2d`, `LeakyReLU`, and a Gradient Penalty mechanism to enforce 1-Lipschitz continuity for highly stable training.

## Dataset Reference

This project uses **PathMNIST** from MedMNIST v2.
Citation: *Jiancheng Yang, Rui Shi, Donglai Wei, et al. "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data (2023).*

## Limitations and Notes

While the tool is designed for practical use and generating medical imagery, please be aware of the following limitations:

- **Model Trained on Specific Dataset**
  The model was trained exclusively on the PathMNIST dataset (colon pathology). It may not generalize well to other types of medical imaging (e.g., MRI, X-Ray) or other tissue types not present in the dataset.
- **Resolution Limit**
  The generated images are fixed at 64x64 resolution, which corresponds to the standard PathMNIST image dimensions preprocessing steps.
- **No Graphical Interface (GUI)**
  The tool is built for CLI-based workflows. A GUI is not currently provided.

## License

*(If applicable, specify your license here, e.g., This project is licensed under the MIT License.)*