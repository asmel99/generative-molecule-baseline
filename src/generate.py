import torch
import pandas as pd
from rdkit import Chem

def generate_molecules():
    # Load dataset to rebuild vocabulary context
    df = pd.read_csv("clean_test.csv")
    smiles_list = df['Canonical_SMILES'].tolist()
    
    chars = sorted(list(set("".join(smiles_list))))
    chars = ['<pad>'] + chars  
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    max_len = 50
    latent_dim = 64

    # Re-instantiate Generator Architecture
    class SMILES_Generator(torch.nn.Module):
        def __init__(self, latent_dim, vocab_size, max_len):
            super(SMILES_Generator, self).__init__()
            self.max_len = max_len
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, max_len * vocab_size)
            )
            self.vocab_size = vocab_size

        def forward(self, z):
            batch_size = z.size(0)
            out = self.fc(z)
            out = out.view(batch_size, self.max_len, self.vocab_size)
            return out

    generator = SMILES_Generator(latent_dim, vocab_size, max_len)
    
    # Load trained weights
    try:
        generator.load_state_dict(torch.load("generator.pth"))
        generator.eval()
        print("Loaded generator weights successfully.")
    except FileNotFoundError:
        print("No saved 'generator.pth' found. Please run train_gan.py first.")
        return

    # Sample new molecules
    num_samples = 10
    z_sample = torch.randn(num_samples, latent_dim)

    with torch.no_grad():
        generated_logits = generator(z_sample)
        predicted_indices = torch.argmax(generated_logits, dim=-1)

    generated_smiles = []
    for seq in predicted_indices:
        chars_list = [idx_to_char[idx.item()] for idx in seq]
        smiles_str = "".join(chars_list).replace("<pad>", "")
        generated_smiles.append(smiles_str)

    print("\nGenerated Raw SMILES Strings:")
    for i, s in enumerate(generated_smiles):
        print(f"  {i+1}: {s}")

    # Validate with RDKit
    print("\nValidating with RDKit:")
    valid_count = 0
    for i, s in enumerate(generated_smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            print(f"  Molecule {i+1} ({s}) is VALID! Canonical: {Chem.MolToSmiles(mol)}")
            valid_count += 1
        else:
            print(f"  Molecule {i+1} ({s}) is invalid.")

    print(f"\nResult: {valid_count}/{num_samples} valid molecules generated.")

if __name__ == "__main__":
    generate_molecules()
