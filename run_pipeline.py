import argparse
import sys
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

# Path injection
sys.path.append(os.getcwd())

from src.aligner import Aligner
from src.llm_client import LLMManager
from src.models import SimilarityRefiner
from src.trainer import Stage1_5Dataset, run_training
from src.metrics import evaluate_and_save_metrics
from src.encoders import DATASET_REGISTRY

def parse_args():

    parser = argparse.ArgumentParser(description="SENSE: SEmantic Neural Sparse Extraction Pipeline")
    # Paths and Data
    parser.add_argument("--dataset", type=str, default="imagenet_eeg_test", help="Path to the EEG dataset .pth file")
    parser.add_argument("--vocab_path", type=str, default="data/imagenet_train_corpus.pt", help="Path to the encoded word corpus")
    parser.add_argument("--output_dir", type=str, default="./pipeline_test", help="Where to save outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    
    # Mode Selection
    parser.add_argument("--mode", type=str, choices=["naive", "train", "inference"], default="naive", 
                        help="naive: Cosine Sim only | train: Train MLP | inference: Use trained MLP")
    
    # MLP Config
    parser.add_argument("--loss", type=str, choices=["bce", "focal", "contrastive"], default="bce")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth weights for inference")
    parser.add_argument("--eeg_encoder", type=str, default="channelnet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # LLM & Eval Logic
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    dataset_name = os.path.basename(args.dataset).replace('.pth', '')
    
    # 1. EEG Encoding Step (Assumes raw EEG -> CLIP latents)
    # This logic checks for existing latents to save time/compute
    clip_latents_path = os.path.join(args.output_dir, f"{dataset_name}_pipeline_test_latents.pt")
    
    if not os.path.exists(clip_latents_path):
        print(f"--- Step 1: Encoding EEG via {args.eeg_encoder} ---")
        from src.encoders import process_channelnet
        process_channelnet(args.dataset, clip_latents_path, args.device, args.batch_size)
    else:
        print(f"--- Found existing latents at {clip_latents_path} ---")

    # 2. Alignment Logic (The Core Switch)
    final_alignment_path = ""

    dataset_path = DATASET_REGISTRY.get(args.dataset, None)
    print(f"Dataset path for training: {dataset_path}")

    if args.mode == "train":
        print(f"--- Mode: Training MLP ({args.loss} loss) ---")
        train_ds = Stage1_5Dataset(clip_latents_path, args.vocab_path)
        loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        # Initialize model: Contrastive loss uses 'NoScaling' (use_scaling=False)
        model = SimilarityRefiner(train_ds.vocab_embeddings, use_scaling=(args.loss != "contrastive"))
        
        save_name = f"mlp_{args.eeg_encoder}_{args.loss}_{args.epochs}eps.pth"
        save_path = os.path.join("checkpoints", save_name)
        
        run_training(model, loader, args.device, args.epochs, args.loss, save_path)
        print(f"Training complete. Model saved to {save_path}. Exiting.")
        return # Training usually stops here before inference

    elif args.mode == "inference":
        print(f"--- Mode: MLP Inference using {args.checkpoint} ---")
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            raise ValueError("Inference mode requires a valid --checkpoint path.")

        vocab_info = torch.load(args.vocab_path)
        model = SimilarityRefiner(vocab_info["embeddings"])
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        model.to(args.device).eval()
        
        latent_dataset = torch.load(clip_latents_path)
        aligned_results = []

        for item in tqdm(latent_dataset, desc="MLP Mapping"):
            eeg_vec = item['eeg_clip_latent'].to(args.device).float()
            if eeg_vec.dim() == 1: eeg_vec = eeg_vec.unsqueeze(0)

            with torch.no_grad():
                logits, refined_latent = model(eeg_vec)
                probs = torch.sigmoid(logits).squeeze()
            
            scores, indices = probs.topk(min(args.top_k, len(vocab_info["words"])))
            bow = [{"word": vocab_info["words"][idx], "score": s.item()} for s, idx in zip(scores, indices)]

            aligned_results.append({
                "subject": item.get("subject"),
                "gt_object_label": item.get("object_label", ""),
                "gt_caption": item.get("caption", ""),
                "predicted_object_label": item.get("predicted_object_label", "n/a"),
                "prediction_confidence": item.get("prediction_confidence", 0.0),
                "bow": bow,
                "prompt_words": [w['word'] for w in bow],
                "refined_latent": refined_latent.cpu()
            })
        
        final_alignment_path = os.path.join(args.output_dir, f"{dataset_name}_mlp_{args.loss}_aligned.pt")
        torch.save(aligned_results, final_alignment_path)

    elif args.mode == "naive":
        print("--- Mode: Naive Global Alignment ---")
        global_aligner = Aligner(args.vocab_path, device=args.device)
        latent_dataset = torch.load(clip_latents_path)
        aligned_results = []

        for item in tqdm(latent_dataset, desc="Naive Aligning"):
            bow = global_aligner.align(
                item['eeg_clip_latent'], 
                top_k=args.top_k
            )
            aligned_results.append({
                "subject": item.get("subject"),
                "gt_object_label": item.get("object_label", ""),
                "gt_caption": item.get("caption", ""),
                "predicted_object_label": item.get("predicted_object_label", "n/a"),
                "prediction_confidence": item.get("prediction_confidence", 0.0),
                "bow": bow,
                "prompt_words": [w['word'] for w in bow]
            })
        
        final_alignment_path = os.path.join(args.output_dir, f"{dataset_name}_naive_aligned.pt")
        torch.save(aligned_results, final_alignment_path)

    # 3. Semantic Decoding via LLM
    if not args.skip_llm and final_alignment_path:
        print(f"--- Step 3: LLM Caption Generation ---")
        llm_manager = LLMManager(provider="openai", model_name="gpt-4o-mini")
        
        final_gen_path = final_alignment_path.replace(".pt", "_captions.pt")
        llm_manager.run_decoding_experiment(
            input_path=final_alignment_path,
            output_path=final_gen_path
        )

    final_csv_path = "results/gemini/mlp_test_llm_gemini_focal_loss_learnt_scaling.csv"
    # 4. Evaluation
    if not args.skip_eval and final_csv_path:
        print(f"--- Step 4: Running Metrics ---")
        evaluate_and_save_metrics(final_csv_path, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
