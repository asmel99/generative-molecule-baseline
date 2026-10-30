# Generative Molecule Baseline (GAN)

This repository contains a deep learning baseline using a **Generative Adversarial Network (GAN)** built with PyTorch and RDKit. It is designed to learn the chemical grammar of SMILES strings and generate novel molecular structures for industrial bioactive discovery.

## 🔬 Research Context
- **Institution:** FST Marrakech (Université Cadi Ayyad)
- **Project Topic:** *Intégration de l’informatique et de l’intelligence Artificielle pour la Découverte de Nouvelles Molécules Bioactives à Visée Industrielle*

## 🚀 Tech Stack
- **Python 3**
- **PyTorch** (Deep Learning & GAN Architecture)
- **RDKit** (Cheminformatics & Molecular Validation)
- **Pandas / NumPy** (Data Processing)

## 📁 Repository Structure
- `src/model.py`: Generator and Discriminator neural network architectures.
- `src/train_gan.py`: Training loop tracking adversarial losses.
- `src/generate.py`: Sampling and RDKit validity checking for generated molecules.
