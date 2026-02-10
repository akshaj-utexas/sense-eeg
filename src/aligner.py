import torch

class Aligner:
    def __init__(self, corpus_path, device="cuda"):
        self.device = device
        data = torch.load(corpus_path)
        self.words = data["words"]
        self.word_embs = data["embeddings"].to(device) # Already normalized

    @torch.no_grad()
    def align(self, eeg_latent, global_noise, top_k=15):
        """
        eeg_latent: [1, 128, 768]
        Returns: List of {"word": str, "score": float}
        """
        # 1. Mean pool time dimension and Center
        vec = eeg_latent.to(self.device).mean(dim=1) 
        vec = vec - global_noise
        vec /= vec.norm(dim=-1, keepdim=True)
        
        # 2. Similarity Search
        sims = (vec @ self.word_embs.T).squeeze(0)
        scores, indices = sims.topk(top_k)
        
        return [{"word": self.words[idx.item()], "score": round(score.item(), 4)} 
                for score, idx in zip(scores, indices)]

def calculate_noise(dataset, device):
    all_vecs = [item['eeg_clip_latent'].mean(dim=1) for item in dataset]
    return torch.cat(all_vecs).mean(dim=0, keepdim=True).to(device)