# Art Detectors Pipeline

A unified pipeline for AI-generated image detection, style analysis, captioning, and restyling with automated data collection and continuous model evaluation.

## Project Overview

This project implements an end-to-end pipeline for detecting and analyzing AI-generated images from Reddit. The system combines multiple state-of-the-art models (CLIP, BLIP, SuSy) with custom transfer learning to classify images as authentic human art, DALL-E generated, or MidJourney generated.

### Architecture
```
Reddit Data Collection → BigQuery Storage → AI Detection Pipeline → Results Analysis
         ↓                      ↓                    ↓                      ↓
   [Apache Airflow]      [image_metadata]    [CLIP + BLIP + SuSy]   [predictions]
```

### Components

1. **Data Collection** (`get_new_data_apache.py`, `extract_data.ipynb`)
   - Automated Reddit scraping using PRAW
   - Collects images from r/dalle2, r/midjourney, r/aiArt, and r/Art
   - Filters by flair and content type (images only)
   - Stores metadata in BigQuery
   - Scheduled via Apache Airflow for continuous data ingestion

2. **AI Detection Pipeline** (`get_prediction.py`)
   - Downloads images from BigQuery metadata
   - Runs multi-model analysis:
     - **CLIP**: Top-5 artistic style predictions
     - **BLIP**: Natural language image captioning
     - **SuSy (Transfer Learning)**: 3-class AI detection
   - Stores results in BigQuery and CSV
   - Evaluates model accuracy against ground truth labels

3. **Transfer Learning Model**
   - Fine-tuned SuSy model for 3-class classification
   - Trained on 1000 images per class (WikiArt, MidJourney, DALL-E 3)
   - Two-stage training: projection layer → full model fine-tuning

### Data Pipeline Flow
```
1. Subreddit Monitoring
   ├── r/dalle2 → ~0.38 posts/day (filtered)
   ├── r/midjourney → ~6.62 posts/day (filtered)
   ├── r/aiArt → ~53.56 posts/day (filtered)
   └── r/Art → ~838 posts/1000 (baseline authentic images)

2. Data Storage (BigQuery)
   ├── image_metadata table
   │   ├── submission_id
   │   ├── filename
   │   ├── url
   │   ├── subreddit
   │   ├── date
   │   └── model (ground truth label)
   └── predictions table
       ├── submission_id
       ├── model (ground truth)
       ├── BLIP_result
       ├── CLIP_top_1-5_mod + probs
       ├── SuSy_model (prediction)
       ├── SuSy_correct (accuracy flag)
       ├── SuSy_dalle/midjourney/authentic_prob
       └── created_at

3. Prediction Pipeline
   ├── Query new submissions (5 per model)
   ├── Download images to prediction/images/
   ├── Run inference (CLIP + BLIP + SuSy)
   ├── Save results to BigQuery
   └── Export to CSV (prediction/results/)

4. Continuous Evaluation
   └── Compare SuSy predictions vs ground truth labels
       └── Track accuracy per model class
```

### Key Features

- **Automated Data Collection**: Apache Airflow schedules Reddit scraping and BigQuery updates
- **Multi-Model Analysis**: Combines CLIP, BLIP, and custom SuSy models
- **Ground Truth Validation**: Tracks prediction accuracy using subreddit labels
- **Scalable Storage**: BigQuery handles metadata and predictions with SQL queries
- **Dual Output**: Results saved to both BigQuery (analytics) and CSV (portability)

## Features

- **Style Detection**: CLIP-based artistic style classification
- **Image Captioning**: BLIP-generated natural language descriptions
- **AI Detection**: SuSy-based authenticity classification
  - Original: 6-class classification (authentic, DALL-E 3, Stable Diffusion, MidJourney V5/V6, etc.)
  - Transfer Learning: 3-class classification (authentic, MidJourney, DALL-E 3)
- **Image Restyling**: Stable Diffusion XL img2img transformation

## Installation
```bash
# Clone the repository
git clone https://github.com/gracey0630/eecs6893-final-project.git
cd eecs6893-final-project

# Install in editable mode
pip install -e .

# Additional dependencies for data collection
pip install praw google-cloud-bigquery pandas requests
```

## Project Structure
```
eecs6893-final-project/
├── src/
│   └── artdetectors/
│       ├── pipeline.py          # Main detection pipeline
│       ├── susy_transfer.py     # Transfer learning model
│       ├── clip_style.py        # CLIP style predictor
│       ├── blip_caption.py      # BLIP captioner
│       ├── restyle.py           # Image restyling
│       ├── models/
│       │   └── best_model_stage2.pth
│       └── data/
│           ├── style.txt
│           └── style_clip_features.pt
├── model_test_results           # Results from model tested against Reddit data
├── example.py                   # Example code for the artdetector
├── example_images               # Example images for the artdetector
├── data/
│   └── get_new_data_apache.py       # Reddit scraper (Airflow DAG)
│   └── get_new_data_local.py        # Reddit scraper (for local testing)
│   └── extract_data.ipynb           # Data analysis notebook
├── get_prediction.py            # Prediction pipeline
└── prediction/
    ├── images/                  # Downloaded images
    └── results/                 # CSV outputs
```

## Data Collection Setup

### 1. Configure Reddit API

Create a Reddit app at https://www.reddit.com/prefs/apps and add credentials:
```python
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_SECRET",
    user_agent="YOUR_USER_AGENT"
)
```

### 2. Setup BigQuery
```bash
# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

# Update project settings in get_prediction.py
PROJECT_ID = "your-project-id"
DATASET_ID = "reddit_scrape"
```

### 3. Run Data Collection
```bash
# One-time: Collect initial dataset
jupyter notebook extract_data.ipynb

# Continuous: Schedule with Airflow
python get_new_data_apache.py
```

### 4. Run Prediction Pipeline
```bash
# Process new images and generate predictions
python get_prediction.py
```

**Output:**
- BigQuery table: `predictions` (analytics-ready)
- CSV files: `prediction/results/all_predictions.csv` (export)
- Images: `prediction/images/{submission_id}.jpg`

## Quick Start

### Basic Usage (6-class original SuSy)
```python
from artdetectors import ImageAnalysisPipeline

# Initialize pipeline
pipe = ImageAnalysisPipeline()

# Analyze an image
result = pipe.analyze("image.jpg")

print("Styles:", result["styles"])
print("Caption:", result["caption"])
print("AI Detection:", result["susy"])
```

**Output:**
```python
{
  "styles": [
    {"style": "impressionism", "score": 0.85},
    {"style": "oil painting", "score": 0.72},
    ...
  ],
  "caption": "a woman standing in a field of flowers",
  "susy": {
    "pred_class": "dalle-3-images",
    "confidence": 0.89,
    "probs": {
      "authentic": 0.05,
      "dalle-3-images": 0.89,
      "diffusiondb": 0.01,
      "midjourney-images": 0.03,
      "midjourney_tti": 0.01,
      "realisticSDXL": 0.01
    },
    "model_type": "original_6class"
  }
}
```

### Transfer Learning Model (3-class)

Our fine-tuned transfer learning model provides simplified 3-class classification:
```python
from artdetectors import ImageAnalysisPipeline

# Use transfer learning model (automatically downloads from Hugging Face)
pipe = ImageAnalysisPipeline(use_transfer_learning=True)

result = pipe.analyze("image.jpg")
print(result["susy"])
```

**Output:**
```python
{
  "pred_class": "midjourney",
  "confidence": 0.94,
  "probs": {
    "authentic": 0.03,
    "midjourney": 0.94,
    "dalle3": 0.03
  },
  "model_type": "transfer_learning_3class"
}
```

### Image Restyling

Transform images into different artistic styles:
```python
from artdetectors import ImageAnalysisPipeline

pipe = ImageAnalysisPipeline(enable_restyler=True)

# Restyle an image
output = pipe.restyle_image(
    "photo.jpg",
    target_style="impressionism",
    strength=0.5,
    guidance_scale=5.0,
    num_inference_steps=30
)

# Save the restyled image
output["restyled_image"].save("restyled.png")
print("Caption used:", output["caption_used"])
```

## API Reference

### `ImageAnalysisPipeline`

#### Initialization Parameters
```python
ImageAnalysisPipeline(
    style_txt_path=None,              # Path to style definitions
    style_features_cache=None,        # Path to cached CLIP features
    susy_repo_id="HPAI-BSC/SuSy",    # HuggingFace repo for SuSy
    susy_filename="SuSy.pt",         # SuSy model filename
    device="auto",                    # Device: "cuda", "cpu", "mps", or "auto"
    enable_restyler=True,            # Enable SDXL restyling
    use_transfer_learning=False,     # Use 3-class transfer learning model
    transfer_checkpoint=None         # Custom checkpoint path (optional)
)
```

#### Methods

##### `analyze(image, topk_styles=5, caption_prompt=None)`

Fast analysis using detectors only.

**Parameters:**
- `image`: Image path (str/Path) or PIL Image
- `topk_styles`: Number of top styles to return (default: 5)
- `caption_prompt`: Optional prompt for BLIP captioning

**Returns:**
```python
{
    "styles": [{"style": str, "score": float}, ...],
    "caption": str,
    "susy": {
        "pred_class": str,
        "confidence": float,
        "probs": dict,
        "model_type": str
    }
}
```

##### `restyle_image(image, target_style, ...)`

Generate a restyled version of the image.

**Parameters:**
- `image`: Image path (str/Path) or PIL Image
- `target_style`: Target artistic style (str)
- `caption_prompt`: Optional caption prompt
- `negative_prompt`: Negative prompt for generation
- `strength`: Transformation strength (0.0-1.0, default: 0.3)
- `guidance_scale`: Guidance scale (default: 5.0)
- `num_inference_steps`: Number of diffusion steps (default: 30)
- `seed`: Random seed for reproducibility

**Returns:**
```python
{
    "restyled_image": PIL.Image,
    "caption_used": str
}
```

## Transfer Learning Model

### About

The transfer learning model is a fine-tuned version of SuSy that classifies images into 3 categories:
- **Authentic**: Real, human-created images
- **MidJourney**: Images generated by MidJourney (V5/V6)
- **DALL-E 3**: Images generated by DALL-E 3

### Training Details

- **Base Model**: SuSy (HPAI-BSC/SuSy)
- **Training Data**: 1000 images per class
  - Authentic: WikiArt (human-created digital art)
  - MidJourney: Generated samples from r/midjourney
  - DALL-E 3: Generated samples from r/dalle2
- **Architecture**: Projection layer (6→3 classes)
- **Training Strategy**: Two-stage
  - Stage 1: Train projection layer only (10 epochs)
  - Stage 2: Fine-tune entire model (10 epochs)
- **Performance** (Evaluation on Reddit test data):

| Class/Model | Original SuSy | Finetuned SuSy |
|-------------|---------------|----------------|
| Authentic   | 68.22%        | 57.71%         |
| DALLE       | 0.32%         | 14.38%         |
| Midjourney  | 0.28%         | 43.37%         |

**Key Insight**: The transfer learning model significantly improves detection of AI-generated images (DALLE: 0.32% → 14.38%, MidJourney: 0.28% → 43.37%) compared to the original 6-class SuSy model, demonstrating the value of fine-tuning on domain-specific data.

### Using a Custom Checkpoint
```python
from artdetectors import ImageAnalysisPipeline

# Use a local checkpoint
pipe = ImageAnalysisPipeline(
    use_transfer_learning=True,
    transfer_checkpoint="path/to/your/best_model_stage2.pth"
)
```

### Model Location

The transfer learning model is automatically downloaded from Hugging Face:
- Repository: `your-username/susy-transfer-3class`
- File: `best_model_stage2.pth`

## Data Collection Details

### Reddit Subreddits

- **r/dalle2**: DALL-E 2 and DALL-E 3 generated images
- **r/midjourney**: MidJourney V5/V6 AI-generated art
- **r/aiArt**: Mixed AI-generated art (various models)
- **r/Art**: Human-created authentic art (baseline)

### Collection Statistics

Initial collection (November 2024):
- DALL-E: 291 images (0.38 posts/day)
- MidJourney: 192 images (6.62 posts/day)
- AI Art: 482 images (53.56 posts/day)
- Art: 838 images (baseline)

### Filtering Criteria

- Images only (no text posts)
- Specific flair tags (AI model identifiers)
- SFW content only
- Valid image formats (.jpg, .jpeg, .png)

## Advanced Usage

### Batch Processing
```python
from pathlib import Path
from artdetectors import ImageAnalysisPipeline

pipe = ImageAnalysisPipeline(use_transfer_learning=True)

image_dir = Path("images")
results = []

for img_path in image_dir.glob("*.jpg"):
    result = pipe.analyze(str(img_path))
    results.append({
        "filename": img_path.name,
        "class": result["susy"]["pred_class"],
        "confidence": result["susy"]["confidence"]
    })

# Print summary
for r in results:
    print(f"{r['filename']}: {r['class']} ({r['confidence']:.2%})")
```

### Confidence Thresholding
```python
from artdetectors import ImageAnalysisPipeline

pipe = ImageAnalysisPipeline(use_transfer_learning=True)
result = pipe.analyze("image.jpg")

confidence = result["susy"]["confidence"]

if confidence > 0.85:
    print("✓ High confidence prediction")
elif confidence > 0.65:
    print("⚠ Medium confidence - review recommended")
else:
    print("❌ Low confidence - manual review required")
    print("Probabilities:", result["susy"]["probs"])
```

### Comparing Models
```python
from artdetectors import ImageAnalysisPipeline

# Original 6-class model
pipe_original = ImageAnalysisPipeline(use_transfer_learning=False)
result_original = pipe_original.analyze("image.jpg")

# Transfer learning 3-class model
pipe_transfer = ImageAnalysisPipeline(use_transfer_learning=True)
result_transfer = pipe_transfer.analyze("image.jpg")

print("Original (6 classes):", result_original["susy"]["pred_class"])
print("Transfer (3 classes):", result_transfer["susy"]["pred_class"])
```

## Hardware Requirements

### Minimum
- **GPU**: Not required (CPU/MPS supported)
- **RAM**: 8GB
- **Storage**: 5GB for models

### Recommended
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for restyling)
- **RAM**: 16GB+
- **Storage**: 10GB+

### Device Selection
```python
# Automatic (recommended)
pipe = ImageAnalysisPipeline(device="auto")

# Manual
pipe = ImageAnalysisPipeline(device="cuda")  # NVIDIA GPU
pipe = ImageAnalysisPipeline(device="mps")   # Apple Silicon
pipe = ImageAnalysisPipeline(device="cpu")   # CPU only
```

## Troubleshooting

### Model Download Issues

If the model fails to download from Hugging Face:
```python
from huggingface_hub import hf_hub_download

# Download manually
model_path = hf_hub_download(
    repo_id="your-username/susy-transfer-3class",
    filename="best_model_stage2.pth",
    cache_dir="./cache"
)

# Use the downloaded model
pipe = ImageAnalysisPipeline(
    use_transfer_learning=True,
    transfer_checkpoint=model_path
)
```

### Out of Memory

Reduce batch size or disable restyler:
```python
pipe = ImageAnalysisPipeline(enable_restyler=False)
```

### Slow Performance

- Use GPU if available
- Reduce `num_inference_steps` for restyling
- Disable restyler for detection-only use

## Citation

If you use this pipeline, please cite:
```bibtex
@software{artdetectors2024,
  title={Art Detectors: AI-Generated Image Detection Pipeline},
  author={Yoon, GaHyun and Shinde, Abhitay and Okubo, Christine},
  year={2024},
  url={https://github.com/gracey0630/eecs6893-final-project}
}
```

Original SuSy model:
```bibtex
@misc{bernabeu2024susy,
    title={Present and Future Generalization of Synthetic Image Detectors}, 
    author={Pablo Bernabeu-Perez and Enrique Lopez-Cuena and Dario Garcia-Gasulla},
    year={2024},
    eprint={2409.14128},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License

Apache 2.0 (same as original SuSy model)

## Authors

- GaHyun Yoon
- Abhitay Shinde
- Christine Okubo

## Acknowledgments

- [SuSy Model](https://huggingface.co/HPAI-BSC/SuSy) by HPAI-BSC
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [BLIP](https://github.com/salesforce/BLIP) by Salesforce
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) by Stability AI
- Reddit communities: r/dalle2, r/midjourney, r/aiArt, r/Art