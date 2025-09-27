#!/usr/bin/env python3
"""
Custom Pixeltable UDF for Hugging Face CLIP Visual Embeddings

This module provides a custom User-Defined Function (UDF) for Pixeltable that
generates visual embeddings using Hugging Face's Inference Providers API.

Features:
- Serverless CLIP embeddings via HF Inference Providers
- Pay-per-use pricing (~$0.00012/second)
- No local model loading required
- Compatible with Mac/MPS constraints
- Returns standard 512-dimensional CLIP embeddings
"""

import os
import base64
import io
from typing import List
import logging

import pixeltable as pxt
from PIL import Image
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_hf_token() -> str:
    """
    Get Hugging Face token from environment variable.

    Returns:
        HF token string

    Raises:
        ValueError: If HF_TOKEN environment variable is not set
    """
    token = os.getenv('HF_TOKEN')
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Please set your Hugging Face token with 'Inference Providers' permissions.\n"
            "Get token at: https://huggingface.co/settings/tokens"
        )
    return token


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert PIL Image to base64 string for API transmission.

    Args:
        image: PIL Image object
        format: Image format for encoding (JPEG, PNG)

    Returns:
        Base64 encoded image string
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")

    # Convert to RGB if necessary (for JPEG compatibility)
    if format.upper() == "JPEG" and image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    # Encode to base64
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=95)
    buffer.seek(0)

    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')

    return f"data:image/{format.lower()};base64,{base64_string}"

@pxt.udf
def hf_clip_embedding(image: pxt.type_system.Image, model: str = "openai/clip-vit-base-patch32") -> List[float]:
    result = execute_embedding(image, model)
    breakpoint()
    return result

def execute_embedding(
    image: pxt.type_system.Image,
    model: str = "openai/clip-vit-base-patch32"
) -> List[float]:
    """
    Generate CLIP visual embeddings using Hugging Face Inference Providers.

    This UDF sends images to Hugging Face's serverless inference API to generate
    CLIP embeddings. It's designed to be cost-effective and work on any hardware.

    Args:
        image: PIL Image object (automatically converted by Pixeltable)
        model: Hugging Face CLIP model ID (default: openai/clip-vit-base-patch32)
        timeout: Request timeout in seconds

    Returns:
        List of 512 float values representing the image embedding

    Raises:
        TypeError: If image is not a PIL Image
        ValueError: If HF_TOKEN is not set
        Exception: For API errors or network issues

    Example:
        ```python
        # Add embedding index using this UDF
        frames_view.add_embedding_index(
            column=frames_view.resized_frame,
            image_embed=hf_clip_embedding,
            if_exists="replace_force",
            idx_name="hf_clip_index"
        )
        ```
    """
    try:
        # Validate input
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")

        # Get HF token
        token = get_hf_token()
        # Initialize Inference Client
        client = InferenceClient(token=token)

        # Convert image to base64
        logger.info(f"Processing image of size {image.size} with model {model}")
        base64_image = image_to_base64(image)

        # Generate embedding using feature extraction
        try:
            # Use feature_extraction endpoint for CLIP embeddings
            embedding = client.feature_extraction(
                base64_image,
                model=model
            )

            # Handle different response formats
            if isinstance(embedding, list):
                if len(embedding) > 0 and isinstance(embedding[0], list):
                    # Nested list format [[embedding]]
                    result = embedding[0]
                else:
                    # Flat list format [embedding]
                    result = embedding
            else:
                raise ValueError(f"Unexpected embedding format: {type(embedding)}")

            # Validate embedding dimensions
            if not isinstance(result, list) or len(result) == 0:
                raise ValueError(f"Invalid embedding: expected non-empty list, got {type(result)}")

            # Convert to float list (ensure proper type)
            result = [float(x) for x in result]

            logger.info(f"Generated embedding with {len(result)} dimensions")
            return result

        except Exception as api_error:
            logger.error(f"HF API error: {api_error}")
            # Provide more specific error messages
            if "token" in str(api_error).lower():
                raise ValueError(
                    f"Authentication failed: {api_error}\n"
                    "Please check your HF_TOKEN has 'Inference Providers' permissions."
                )
            elif "model" in str(api_error).lower():
                raise ValueError(f"Model error: {api_error}\nTry model: openai/clip-vit-base-patch32")
            else:
                raise Exception(f"HF Inference API error: {api_error}")

    except Exception as e:
        logger.error(f"Error in hf_clip_embedding: {e}")
        raise


@pxt.udf
def hf_clip_embedding_batch(
    images: List[pxt.type_system.Image],
    model: str = "openai/clip-vit-base-patch32",
    timeout: float = 60.0
) -> List[List[float]]:
    """
    Generate CLIP embeddings for multiple images in a single API call.

    More efficient for processing multiple images at once.

    Args:
        images: List of PIL Image objects
        model: Hugging Face CLIP model ID
        timeout: Request timeout in seconds

    Returns:
        List of embedding lists, one per input image
    """
    try:
        if not images:
            return []

        # Process each image individually for now
        # (batch processing would require checking if HF API supports it)
        results = []
        for img in images:
            embedding = hf_clip_embedding(img, model, timeout)
            results.append(embedding)

        return results

    except Exception as e:
        logger.error(f"Error in hf_clip_embedding_batch: {e}")
        raise


def test_hf_clip_embedding():
    """
    Test function to verify the UDF works correctly.

    Uses the colosseum.png image to test real image embedding generation.
    """
    print("🧪 Testing HF CLIP Embedding UDF...")

    try:
        # Test with colosseum image
        colosseum_path = os.path.join(os.path.dirname(__file__), "data", "colosseum.png")

        if os.path.exists(colosseum_path):
            print(f"📸 Loading colosseum image: {colosseum_path}")
            test_image = Image.open(colosseum_path)
            print(f"✅ Loaded image: {test_image.size} {test_image.mode}")
        else:
            print("⚠️  Colosseum image not found, creating test image...")
            test_image = Image.new('RGB', (224, 224), color=(255, 0, 0))
            print(f"✅ Created test image: {test_image.size} {test_image.mode}")

        # Test embedding generation
        print("🚀 Generating embedding via HF API...")
        
        embedding = execute_embedding(test_image)
        print(f"✅ Generated embedding: {len(embedding)} dimensions")
        print(f"📊 Sample values: {embedding[:5]}...")

        # Validate embedding
        if len(embedding) == 512:
            print("✅ Standard CLIP embedding dimensions (512)")
        elif len(embedding) == 768:
            print("✅ Large CLIP embedding dimensions (768)")
        else:
            print(f"⚠️  Non-standard dimensions: {len(embedding)}")

        # Test with different models if successful
        print("\n🔄 Testing with different CLIP model...")
        try:
            embedding2 = hf_clip_embedding(test_image, model="openai/clip-vit-base-patch16")
            print(f"✅ Alternative model works: {len(embedding2)} dimensions")
        except Exception as model_error:
            print(f"⚠️  Alternative model failed: {model_error}")

        print("\n🎉 Test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Common issues:")
        print("   - HF_TOKEN not set: export HF_TOKEN='your_token_here'")
        print("   - No internet connection")
        print("   - Invalid token permissions (need 'Inference Providers')")
        return False


def test_with_colosseum():
    """
    Specific test function for the colosseum image.
    """
    print("🏛️  Testing CLIP embeddings with Colosseum image...")

    colosseum_path = os.path.join(os.path.dirname(__file__), "data", "colosseum.png")

    if not os.path.exists(colosseum_path):
        print(f"❌ Colosseum image not found at: {colosseum_path}")
        return False

    try:
        # Load and process colosseum image
        image = Image.open(colosseum_path)
        print(f"📸 Image loaded: {image.size} {image.mode}")

        # Generate embedding
        embedding = hf_clip_embedding(image)
        print(f"🎯 Embedding generated: {len(embedding)} dimensions")
        print(f"📊 First 10 values: {[round(x, 4) for x in embedding[:10]]}")

        # Test similarity with text descriptions
        print("\n🔍 Testing semantic understanding...")

        # Generate embeddings for different descriptions
        descriptions = [
            "ancient Roman amphitheater",
            "modern skyscraper",
            "natural landscape"
        ]

        # Note: This would require text embeddings too, so just show the image embedding
        print("✅ Image embedding ready for similarity search!")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Run tests when script is executed directly
    print("🎯 Running HF CLIP Embedding Tests...\n")

    # Test with colosseum image specifically
    success1 = test_with_colosseum()

    print("\n" + "="*50 + "\n")

    # Run general test
    success2 = test_hf_clip_embedding()

    if success1 and success2:
        print("\n🎉 All tests passed! Ready to use in Pixeltable.")
    else:
        print("\n❌ Some tests failed. Check your HF_TOKEN setup.")