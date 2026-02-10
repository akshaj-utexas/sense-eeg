import torch
import torch.nn as nn
from tqdm import tqdm
import sys

MODEL_REGISTRY = {
    "dream_diffusion": {"checkpoint": "models/dreamdiffusion_checkpoint.pth"}
    # Add more encoders later
}

# We assume sc_mbm is in your PYTHONPATH or the project root
try:
    from sc_mbm.mae_for_eeg import eeg_encoder 
except ImportError:
    print("Error: Ensure 'sc_mbm' is in your Python path.")

class DreamDiffusionPipeline(nn.Module):
    """Fuses MAE Encoder + Projection Layer into one unit."""
    def __init__(self, checkpoint_path, device):
        super().__init__()
        self.device = device
        
        # 1. Setup MAE Encoder (1024-D)
        eeg_params = {
            'time_len': 512, 'patch_size': 4, 'embed_dim': 1024,
            'in_chans': 128, 'depth': 24, 'num_heads': 16, 'global_pool': False
        }
        self.mae_encoder = eeg_encoder(**eeg_params).to(device)
        self.projector = nn.Linear(1024, 768).to(device)
        
        sd = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
        mae_sd = {k.replace('cond_stage_model.mae.', ''): v for k, v in sd.items() if 'cond_stage_model.mae.' in k}
        self.mae_encoder.load_state_dict(mae_sd, strict=True)
        
        proj_sd = {
            'weight': sd['cond_stage_model.mapping.fc.weight'],
            'bias': sd['cond_stage_model.mapping.fc.bias']
        }
        self.projector.load_state_dict(proj_sd, strict=True)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.projector(self.mae_encoder(x))  # f(g(x))

def process_dream_diffusion(dataset_path, output_path, device):
    cfg = MODEL_REGISTRY["dream_diffusion"]
    pipe = DreamDiffusionPipeline(cfg["checkpoint"], device)
    raw_data = torch.load(dataset_path)
    
    final_dataset = []
    for item in tqdm(raw_data, desc="Encoding EEG to CLIP space"):
        eeg_input = item['eeg_tensor'].to(device)
        
        # Handle Padding (440 -> 512)

        padding_needed = 512 - eeg_input.shape[2]
        if padding_needed > 0:
            eeg_input = nn.functional.pad(eeg_input, (0, padding_needed), 'constant', 0)
            
        eeg_clip_latent = pipe(eeg_input)
        
        final_dataset.append({
            "eeg_clip_latent": eeg_clip_latent.cpu(),
            "caption": item["caption"],
            "subject": item.get("subject"),
            "image_path": item.get("image_path")
        })
        
    torch.save(final_dataset, output_path)
    return output_path
