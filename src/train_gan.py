import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

def train_gan():
    # 1. Load cleaned dataset
    df = pd.read_csv("clean_test.csv")
    smiles_list = df['Canonical_SMILES'].tolist()
    print(f"Loaded {len(smiles_list)} molecules for training.")

    # 2. Build Character Vocabulary
    chars = sorted(list(set("".join(smiles_list))))
    chars = ['<pad>'] + chars  
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    print(f"Vocabulary Size: {vocab_size}")

    # 3. Tokenization & Padding
    max_len = 50  
    def encode_smiles(smiles, max_len, char_to_idx):
        encoded = [char_to_idx.get(ch, 0) for ch in smiles]
        if len(encoded) < max_len:
            encoded = encoded + [0] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        return encoded

    encoded_data = [encode_smiles(s, max_len, char_to_idx) for s in smiles_list]
    X_train = torch.tensor(encoded_data, dtype=torch.long)

    # 4. GAN Architecture Definitions
    class SMILES_Generator(nn.Module):
        def __init__(self, latent_dim, vocab_size, max_len):
            super(SMILES_Generator, self).__init__()
            self.max_len = max_len
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, max_len * vocab_size)
            )
            self.vocab_size = vocab_size

        def forward(self, z):
            batch_size = z.size(0)
            out = self.fc(z)
            out = out.view(batch_size, self.max_len, self.vocab_size)
            return out

    class SMILES_Discriminator(nn.Module):
        def __init__(self, max_len, vocab_size):
            super(SMILES_Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(max_len * vocab_size, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    # 5. Hyperparameters & Initialization
    latent_dim = 64
    epochs = 100  # Adjust as needed (e.g., 500+ for full datasets)
    lr = 0002

    generator = SMILES_Generator(latent_dim, vocab_size, max_len)
    discriminator = SMILES_Discriminator(max_len, vocab_size)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    print("\nStarting GAN Training Loop...")

    for epoch in range(epochs):
        batch_size = X_train.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        discriminator.zero_grad()
        real_input = torch.nn.functional.one_hot(X_train, num_classes=vocab_size).float()
        outputs_real = discriminator(real_input)
        loss_D_real = criterion(outputs_real, real_labels)
        
        z = torch.randn(batch_size, latent_dim)
        fake_logits = generator(z)
        fake_input = torch.softmax(fake_logits, dim=-1)
        outputs_fake = discriminator(fake_input.detach())
        loss_D_fake = criterion(outputs_fake, fake_labels)
        
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()
        
        # Train Generator
        generator.zero_grad()
        outputs_fake = discriminator(fake_input)
        loss_G = criterion(outputs_fake, real_labels)
        loss_G.backward()
        optimizer_G.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    print("\nGAN Training Complete!")
    
    # Save model weights if desired
    torch.save(generator.state_dict(), "generator.pth")
    print("Generator model saved to generator.pth")

if __name__ == "__main__":
    train_gan()
