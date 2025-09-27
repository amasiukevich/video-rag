#!/usr/bin/env python3
"""
Huggingface Model Cache Management and Fast CLIP Benchmark Script

This script provides utilities to:
1. Clear Huggingface model cache safely
2. Download CLIP models with fast image processors
3. Benchmark performance differences
4. Verify fast processor usage
"""

import os
import shutil
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def clear_huggingface_cache(selective: bool = True, confirm: bool = True) -> None:
    """
    Clear Huggingface model cache safely.

    Args:
        selective: If True, only remove model caches (keep datasets)
        confirm: If True, ask for confirmation before deletion
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        print("❌ Huggingface cache directory not found")
        return

    # Calculate current cache size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    size_gb = total_size / (1024**3)

    print(f"📊 Current cache size: {size_gb:.2f} GB")

    if selective:
        model_dirs = list(cache_dir.glob("models--*"))
        print(f"🎯 Found {len(model_dirs)} model caches to remove")
        for model_dir in model_dirs[:5]:  # Show first 5
            print(f"   - {model_dir.name}")
        if len(model_dirs) > 5:
            print(f"   ... and {len(model_dirs) - 5} more")
    else:
        print("💣 NUCLEAR OPTION: Will remove entire cache directory")

    if confirm:
        response = input("\n🤔 Proceed with cache cleanup? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return

    try:
        if selective:
            for model_dir in cache_dir.glob("models--*"):
                print(f"🗑️  Removing: {model_dir.name}")
                shutil.rmtree(model_dir)
        else:
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        print("✅ Cache cleanup completed successfully")

        # Show new size
        if cache_dir.exists():
            new_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            new_size_gb = new_size / (1024**3)
            print(f"📊 New cache size: {new_size_gb:.2f} GB")
            print(f"💾 Space freed: {size_gb - new_size_gb:.2f} GB")

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")


def setup_fast_clip(model_name: str = "openai/clip-vit-base-patch32") -> Tuple[CLIPModel, CLIPProcessor, str]:
    """
    Download and setup CLIP with fast image processors.

    Args:
        model_name: Huggingface model identifier

    Returns:
        Tuple of (model, processor, device)
    """
    print(f"🚀 Setting up CLIP model: {model_name}")

    # Determine best device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS doesn't support float16 well
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"🖥️  Using device: {device} with dtype: {dtype}")

    # Download processor with fast tokenizers
    print("⬇️  Downloading fast processor...")
    processor = CLIPProcessor.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )

    # Download model
    print("⬇️  Downloading model...")
    model = CLIPModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    ).to(device)

    print("✅ CLIP setup completed")
    return model, processor, device


def verify_fast_processor(processor: CLIPProcessor) -> bool:
    """
    Verify that we're using fast processors.

    Args:
        processor: CLIP processor to check

    Returns:
        True if using fast processor
    """
    print("\n🔍 Processor Verification:")
    print(f"   Processor type: {type(processor.tokenizer).__name__}")

    is_fast = getattr(processor.tokenizer, 'is_fast', False)
    print(f"   Is fast: {is_fast}")

    if hasattr(processor.tokenizer, 'backend'):
        print(f"   Backend: {processor.tokenizer.backend}")

    # Check image processor
    print(f"   Image processor: {type(processor.image_processor).__name__}")

    return is_fast


def benchmark_clip_performance(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    num_images: int = 10,
    image_size: Tuple[int, int] = (224, 224)
) -> dict:
    """
    Benchmark CLIP processing performance.

    Args:
        model: CLIP model
        processor: CLIP processor
        device: Device being used
        num_images: Number of test images
        image_size: Size of test images

    Returns:
        Performance metrics
    """
    print(f"\n🏃‍♂️ Running performance benchmark with {num_images} images...")

    # Create test images
    test_images = [
        Image.new('RGB', image_size, color=(i*25 % 255, i*50 % 255, i*75 % 255))
        for i in range(num_images)
    ]

    test_text = ["a photo of a cat", "a beautiful landscape", "a red car"]

    # Warmup
    print("🔥 Warming up...")
    with torch.no_grad():
        inputs = processor(images=test_images[0], text=test_text[0], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model(**inputs)

    # Benchmark image processing
    print("📸 Benchmarking image processing...")
    start_time = time.time()

    with torch.no_grad():
        for img in test_images:
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model.get_image_features(**{k: v for k, v in inputs.items() if 'pixel_values' in k})

    image_time = time.time() - start_time

    # Benchmark text processing
    print("📝 Benchmarking text processing...")
    start_time = time.time()

    with torch.no_grad():
        for text in test_text * (num_images // len(test_text) + 1):
            inputs = processor(text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model.get_text_features(**{k: v for k, v in inputs.items() if 'input_ids' in k or 'attention_mask' in k})

    text_time = time.time() - start_time

    # Combined processing
    print("🔄 Benchmarking combined processing...")
    start_time = time.time()

    with torch.no_grad():
        inputs = processor(images=test_images, text=test_text * (num_images // len(test_text) + 1)[:num_images], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        _ = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds, dim=1)

    combined_time = time.time() - start_time

    metrics = {
        'device': device,
        'num_images': num_images,
        'image_processing_time': image_time,
        'text_processing_time': text_time,
        'combined_processing_time': combined_time,
        'images_per_second': num_images / image_time,
        'texts_per_second': num_images / text_time,
        'combined_per_second': num_images / combined_time
    }

    return metrics


def print_benchmark_results(metrics: dict) -> None:
    """Print formatted benchmark results."""
    print("\n📊 Benchmark Results:")
    print(f"   Device: {metrics['device']}")
    print(f"   Test images: {metrics['num_images']}")
    print(f"   Image processing: {metrics['images_per_second']:.2f} images/sec")
    print(f"   Text processing: {metrics['texts_per_second']:.2f} texts/sec")
    print(f"   Combined processing: {metrics['combined_per_second']:.2f} pairs/sec")
    print(f"   Total time: {metrics['image_processing_time'] + metrics['text_processing_time'] + metrics['combined_processing_time']:.3f}s")


def recommend_clip_models() -> None:
    """Print recommended CLIP models for different use cases."""
    models = {
        "Fast & Efficient": [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16"
        ],
        "High Accuracy": [
            "openai/clip-vit-large-patch14",
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
        ],
        "Multilingual": [
            "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        ],
        "Alternative Vision Models": [
            "facebook/dinov2-base",
            "google/vit-base-patch16-224"
        ]
    }

    print("\n🎯 Recommended CLIP Models:")
    for category, model_list in models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"   • {model}")


def main():
    """Main execution function."""
    print("🤗 Huggingface CLIP Benchmark Tool")
    print("=" * 50)

    while True:
        print("\nSelect an option:")
        print("1. Clear model cache")
        print("2. Setup fast CLIP model")
        print("3. Run performance benchmark")
        print("4. Show recommended models")
        print("5. Full cleanup and setup")
        print("6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            selective = input("Selective cleanup (keep datasets)? (Y/n): ").lower() != 'n'
            clear_huggingface_cache(selective=selective)

        elif choice == "2":
            model_name = input("Model name (default: openai/clip-vit-base-patch32): ").strip()
            if not model_name:
                model_name = "openai/clip-vit-base-patch32"

            try:
                model, processor, device = setup_fast_clip(model_name)
                verify_fast_processor(processor)
                print("✅ Setup completed successfully")
            except Exception as e:
                print(f"❌ Setup failed: {e}")

        elif choice == "3":
            model_name = input("Model name (default: openai/clip-vit-base-patch32): ").strip()
            if not model_name:
                model_name = "openai/clip-vit-base-patch32"

            try:
                model, processor, device = setup_fast_clip(model_name)
                verify_fast_processor(processor)
                metrics = benchmark_clip_performance(model, processor, device)
                print_benchmark_results(metrics)
            except Exception as e:
                print(f"❌ Benchmark failed: {e}")

        elif choice == "4":
            recommend_clip_models()

        elif choice == "5":
            print("🔄 Running full cleanup and setup...")
            clear_huggingface_cache(selective=True, confirm=True)
            model, processor, device = setup_fast_clip()
            verify_fast_processor(processor)
            metrics = benchmark_clip_performance(model, processor, device)
            print_benchmark_results(metrics)

        elif choice == "6":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()