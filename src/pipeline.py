import pandas as pd
from rdkit import Chem

def clean_and_standardize_smiles(input_csv, output_csv, smiles_col="SMILES"):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    initial_count = len(df)
    print(f"Initial row count: {initial_count}")
    
    df = df.dropna(subset=[smiles_col])
    
    def standardize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    
    print("Standardizing SMILES strings using RDKit...")
    df['Canonical_SMILES'] = df[smiles_col].apply(standardize)
    
    valid_count_before = len(df)
    df = df.dropna(subset=['Canonical_SMILES'])
    invalid_count = valid_count_before - len(df)
    print(f"Filtered out {invalid_count} invalid SMILES strings.")
    
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['Canonical_SMILES'])
    duplicates_removed = before_dedup - len(df)
    print(f"Removed {duplicates_removed} duplicate structures.")
    
    df.to_csv(output_csv, index=False)
    print(f"Pipeline complete! Saved cleaned data to {output_csv}. Final row count: {len(df)}")
    
    return df

if __name__ == "__main__":
    # Example usage when running script directly
    clean_and_standardize_smiles("raw_test.csv", "clean_test.csv", smiles_col="SMILES")
