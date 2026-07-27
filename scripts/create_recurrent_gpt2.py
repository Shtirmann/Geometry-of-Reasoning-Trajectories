"""
Create GPT-2 with recurrent blocks as in the paper
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class RecurrentGPT2(nn.Module):
    """
    GPT-2 with recurrent blocks at layers 4, 5-6, 7
    As described in the paper
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Base GPT-2
        self.gpt = GPT2Model(config)
        
        # Recurrent groups
        self.recurrent_groups = {
            4: 'single',      # layer 4 self-loop
            5: 'paired',      # layers 5-6 paired
            6: 'paired',
            7: 'single'       # layer 7 self-loop
        }
        
        # Store layers
        self.layers = self.gpt.h
        
        # Recurrent projections (as in paper)
        self.loop_projections = nn.ModuleDict()
        for layer_idx in [4, 5, 6, 7]:
            self.loop_projections[str(layer_idx)] = nn.Linear(
                config.n_embd, 
                config.n_embd
            )
    
    def forward(self, input_ids, num_loops=8):
        """
        Forward pass with recurrent loops
        """
        # Get hidden states
        outputs = self.gpt(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Apply recurrent loops
        for layer_idx, group_type in self.recurrent_groups.items():
            if group_type == 'single':
                # Single layer loop
                hidden_states = self._apply_single_loop(
                    hidden_states, 
                    layer_idx, 
                    num_loops
                )
            elif group_type == 'paired':
                # Paired layers loop (5-6)
                hidden_states = self._apply_paired_loop(
                    hidden_states, 
                    layer_idx, 
                    num_loops
                )
        
        return hidden_states
    
    def _apply_single_loop(self, hidden_states, layer_idx, num_loops):
        """Apply loop to a single layer"""
        for _ in range(num_loops):
            # Project input
            hidden_states = self.loop_projections[str(layer_idx)](
                hidden_states
            )
            # Apply layer
            hidden_states = self.layers[layer_idx](hidden_states)[0]
        return hidden_states
    
    def _apply_paired_loop(self, hidden_states, layer_idx, num_loops):
        """Apply loop to paired layers (5-6)"""
        for _ in range(num_loops):
            # Layer 5
            hidden_states = self.loop_projections[str(layer_idx)](
                hidden_states
            )
            hidden_states = self.layers[layer_idx](hidden_states)[0]
            
            # Layer 6
            hidden_states = self.loop_projections[str(layer_idx+1)](
                hidden_states
            )
            hidden_states = self.layers[layer_idx+1](hidden_states)[0]
        return hidden_states

def create_model():
    """Create model with paper configuration"""
    config = GPT2Config(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=512,
        vocab_size=50257,
    )
    
    model = RecurrentGPT2(config)
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} params")
    
    return model

if __name__ == "__main__":
    model = create_model()
    print(model)
