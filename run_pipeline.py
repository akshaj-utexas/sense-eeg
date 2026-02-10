import argparse
import sys
from typing import List
import os
from src.encoders import process_dream_diffusion
from src.aligner import Aligner, calculate_noise

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Paths and Data
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the EEG dataset")
    # parser.add_argument("--word_corpus", type=str, required=True, help="Path to the word corpus file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Where to save embeddings and captions")
    
    # Model Config
    parser.add_argument("--eeg_encoder", type=str, default="dream_diffusion", help="Name or path of the EEG encoder model")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    
    # Pipeline Logic
    # parser.add_argument("--llms", nargs="+", default=["gpt-4"], help="LLMs to use (gpt-4, gemini, claude)")
    # parser.add_argument("--top_k", type=int, default=10, help="Number of words to retrieve per EEG segment")
    # parser.add_argument("--skip_eval", action="store_true", help="Skip the metrics calculation step")

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Starting Pipeline with {args.eeg_encoder} ---")

    # 1. & 2. Load Data and Encode EEG -> CLIP Space
    # We combine these because the encoder logic handles the loading/saving internally for now
    clip_latents_path = os.path.join(args.output_dir, f"{args.eeg_encoder}_clip_latents.pt")
    
    if args.eeg_encoder == "dream_diffusion":
        process_dream_diffusion(args.dataset_path, clip_latents_path, args.device)

    # works till here ^^

    # aligner = Aligner(args.word_corpus, device=args.device)
    # dataset = torch.load(clip_latents_path)
    # noise = calculate_noise(dataset, device=args.device)

    # final_results = []
    # for item in dataset:
    #     bow_with_scores = aligner.align(item['eeg_clip_latent'], noise, top_k=args.top_k)
    #     final_results.append({
    #         "subject": item.get["subject"],
    #         "gt_caption": item.get["caption"],
    #         "bow": bow_with_scores  # contains words and their similarity scores as floats
    #         "prompt_words": [w['word'] for w in bow_with_scores]  # Just the words for LLM input
    #     })

    # torch.save(final_results, os.path.join(args.output_dir, "aligned_results.pt"))
    
    
    # 3. Retrieve Words from Corpus (Next step)
    # 4. LLM Generation (Next step)
    # 5. Evaluation (Next step)

if __name__ == "__main__":
    main()