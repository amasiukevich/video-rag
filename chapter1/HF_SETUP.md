# Hugging Face CLIP Embeddings Setup Guide

## 🔑 Authentication Setup

### Step 1: Get Your Hugging Face Token

1. **Visit**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Create New Token**:
   - Click "New token"
   - **Name**: `pixeltable-embeddings` (or any name you prefer)
   - **Type**: Select "Fine-grained" for better security
   - **Permissions**: Enable "Make calls to Inference Providers"
3. **Copy Token**: Save the token securely (you won't be able to see it again)

### Step 2: Set Environment Variable

**Option A: Terminal Session (Temporary)**
```bash
export HF_TOKEN="your_token_here"
```

**Option B: Shell Profile (Persistent)**
```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.bash_profile
echo 'export HF_TOKEN="your_token_here"' >> ~/.zshrc
source ~/.zshrc
```

**Option C: Python Script (For Testing)**
```python
import os
os.environ['HF_TOKEN'] = "your_token_here"
```

### Step 3: Install Required Dependencies

```bash
pip install huggingface-hub
```

### Step 4: Verify Setup

Run the test function to verify everything works:

```python
from hf_clip_embeddings import test_hf_clip_embedding
test_hf_clip_embedding()
```

## 🚀 Usage in Pixeltable

### Import the UDF

```python
from hf_clip_embeddings import hf_clip_embedding
```

### Replace Failed Embedding Call

**❌ Old (Broken) Code:**
```python
frames_view.add_embedding_index(
    column=frames_view.resized_frame,
    image_embed=embeddings.using(model="clip-vit-base-patch32"),  # This fails
    if_exists="replace_force",
    idx_name="resized_frames_index"
)
```

**✅ New (Working) Code:**
```python
frames_view.add_embedding_index(
    column=frames_view.resized_frame,
    image_embed=hf_clip_embedding,  # Use custom UDF
    if_exists="replace_force",
    idx_name="hf_clip_index"
)
```

### Advanced Usage

**Custom Model:**
```python
@pxt.udf
def custom_clip_model(image: pxt.type_system.Image) -> List[float]:
    return hf_clip_embedding(image, model="laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
```

## 💰 Cost Estimation

- **Pricing**: ~$0.00012 per GPU-second
- **Typical Processing**: ~0.1-0.5 seconds per image
- **Cost per Image**: ~$0.000012 - $0.00006
- **1000 Images**: ~$0.012 - $0.06

## 🔧 Troubleshooting

### Authentication Errors
```
Authentication failed: ... Please check your HF_TOKEN has 'Inference Providers' permissions.
```
**Solution**: Verify your token has "Make calls to Inference Providers" permission.

### Model Errors
```
Model error: ... Try model: openai/clip-vit-base-patch32
```
**Solution**: Use a supported CLIP model from Hugging Face Hub.

### Network Errors
```
HF Inference API error: ...
```
**Solution**: Check internet connection and try again. HF has built-in retry logic.

## 📋 Supported Models

### Fast & Efficient
- `openai/clip-vit-base-patch32` (default)
- `openai/clip-vit-base-patch16`

### High Accuracy
- `openai/clip-vit-large-patch14`
- `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`

### Alternative
- `sentence-transformers/clip-ViT-B-32-multilingual-v1`

## 🎯 Benefits

✅ **No Local Setup**: No CUDA, PyTorch, or model downloads needed
✅ **Mac Compatible**: Works perfectly on Mac with MPS
✅ **Serverless**: Pay only when generating embeddings
✅ **Scalable**: Auto-routing to best available providers
✅ **Fast**: Optimized inference endpoints
✅ **Reliable**: Built-in error handling and retries