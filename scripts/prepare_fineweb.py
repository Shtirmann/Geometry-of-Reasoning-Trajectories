"""
Download and prepare FineWeb dataset for training
"""

from datasets import load_dataset
import json
from pathlib import Path

def download_fineweb(split='train', num_samples=100000):
    """
    Download FineWeb dataset
    
    FineWeb: ~15M documents, 44TB of text
    We'll use a small subset for testing
    """
    print("Downloading FineWeb dataset...")
    
    # Streaming mode - doesn't download everything
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        split=split,
        streaming=True
    )
    
    # Take first N samples
    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(example['text'])
        if i % 10000 == 0:
            print(f"Downloaded {i} samples...")
    
    # Save to file
    output_dir = Path('data/fineweb')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'samples.json', 'w') as f:
        json.dump(samples, f)
    
    print(f"Saved {len(samples)} samples to {output_dir / 'samples.json'}")
    return samples

if __name__ == "__main__":
    download_fineweb(num_samples=100000)
