# EEG2Text

This repository contains the infrastructure for converting EEG signals into English text by aligning neural embeddings with the CLIP semantic space. The project bypasses traditional end-to-end LLM fine-tuning by leveraging pre-trained multi-modal encoders and Large Language Models as decoders.

---

## Current Progress

The pipeline is currently functional through the **Data Ingestion** and **Embedding Alignment** phases:

1. **EEG Encoding**: Raw EEG tensors are processed through a Masked Autoencoder (MAE) to extract high-dimensional neural features.
2. **CLIP Mapping**: These features are projected via a learned linear layer into the 768-dimensional CLIP latent space, allowing for direct comparison with text embeddings.
3. **Global Noise Centering**: The system calculates a dataset-wide mean vector to center EEG latents, significantly improving retrieval accuracy by removing common-mode neural noise.

---

## File Structure

The project is organized into a modular hierarchy to allow for easy swapping of encoders and datasets.

```text
.
├── run_pipeline.py           # Main entry point for the end-to-end pipeline
├── README.md                 # Project documentation
├── sc_mbm/                   # Core research code for the DreamDiffusion MAE
│   ├── mae_for_eeg.py        # EEG Encoder architecture
│   ├── trainer.py            # Training logic for the encoder
│   └── utils.py              # Helper functions for sc_mbm
├── src/                      # Source modules for pipeline steps
│   ├── encoders.py           # Logic to manage different EEG encoders (e.g., DreamDiffusion)
│   ├── aligner.py            # Similarity search and noise centering logic
│   ├── build_corpus.py       # One-time script to generate CLIP-encoded word indices
│   ├── llm_client.py         # API interface for GPT-4, Gemini, and Claude
│   └── metrics.py            # NLP evaluation metrics (BLEU, ROUGE, BERTScore)
├── scripts/                  # Standalone utility scripts
│   └── evaluate.py           # Post-generation evaluation script
├── data/                     # Raw EEG datasets and processed word corpora
└── models/                   # Model checkpoints and weights

```

---

## Pipeline Components

### 1. Encoders (`src/encoders.py`)

Responsible for loading model checkpoints and performing the forward pass from raw EEG to 1024-D or 768-D CLIP latents. It handles specific preprocessing like time-dimension padding. For now, only has support for dreamdiffusion

### 2. Aligner (`src/aligner.py`)

Computes the cosine similarity between centered EEG latents and a pre-built word corpus. It outputs a "Bag of Words" (BoW) sorted by semantic relevance.

### 3. Build Corpus (`src/build_corpus.py`)

A utility to extract high-frequency nouns, verbs, and adjectives from a dataset (like ImageNet) and encode them into CLIP space. This serves as the searchable dictionary for the pipeline.

---
## Environment setup (Only DreamDiffusion support right now)

Create and activate conda environment named ```dreamdiffusion``` from the ```env.yaml```
```sh
conda env create -f env.yaml
conda activate dreamdiffusion
```

## Usage

To run the pipeline from start to the current encoding milestone:

```bash
python run_pipeline.py --dataset_path data/your_dataset.pth --output_dir ./data --eeg_encoder dream_diffusion

```

---

**Next Steps**: Implementation of the unified `llm_client.py` to convert retrieved word clusters into coherent captions and integrating the automated evaluation suite.