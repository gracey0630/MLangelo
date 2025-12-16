from datetime import datetime
from pathlib import Path
import json
import logging
import pandas as pd
import requests
from google.cloud import bigquery
from artdetectors import ImageAnalysisPipeline
from PIL import Image
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery configuration
PROJECT_ID = "eecs6893-471617"
DATASET_ID = "reddit_scrape"
PREDICTIONS_TABLE = "predictions"
METADATA_TABLE = "image_metadata"

# Directories
IMAGE_DIR = Path("prediction/images")
RESULTS_DIR = Path("prediction/results")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize BigQuery client
client = bigquery.Client(project=PROJECT_ID)

# Initialize pipeline
pipeline = ImageAnalysisPipeline(
    use_transfer_learning=True,
    enable_restyler=False
)


def create_pred_table():
    """
    Create BigQuery table `predictions` if it doesn't exist
    """
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{PREDICTIONS_TABLE}"
    
    schema = [
        bigquery.SchemaField("submission_id", "STRING"),
        bigquery.SchemaField("model", "STRING"),
        
        bigquery.SchemaField("BLIP_result", "STRING"),
        
        bigquery.SchemaField("CLIP_top_1_mod", "STRING"),
        bigquery.SchemaField("CLIP_top_1_prob", "FLOAT"),
        bigquery.SchemaField("CLIP_top_2_mod", "STRING"),
        bigquery.SchemaField("CLIP_top_2_prob", "FLOAT"),
        bigquery.SchemaField("CLIP_top_3_mod", "STRING"),
        bigquery.SchemaField("CLIP_top_3_prob", "FLOAT"),
        bigquery.SchemaField("CLIP_top_4_mod", "STRING"),
        bigquery.SchemaField("CLIP_top_4_prob", "FLOAT"),
        bigquery.SchemaField("CLIP_top_5_mod", "STRING"),
        bigquery.SchemaField("CLIP_top_5_prob", "FLOAT"),
        
        bigquery.SchemaField("SuSy_model", "STRING"),
        bigquery.SchemaField("SuSy_correct", "BOOLEAN"),
        bigquery.SchemaField("SuSy_dalle_prob", "FLOAT"),
        bigquery.SchemaField("SuSy_midjourney_prob", "FLOAT"),
        bigquery.SchemaField("SuSy_authentic_prob", "FLOAT"),
        
        bigquery.SchemaField("created_at", "TIMESTAMP"),
    ]
    
    table = bigquery.Table(table_id, schema=schema)
    
    try:
        table = client.create_table(table)
        logger.info(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")
    except Exception as e:
        if "Already Exists" in str(e):
            logger.info(f"Table {table_id} already exists")
        else:
            logger.error(f"Error creating table: {e}")
            raise


def get_new_submissions():
    """
    Load top 5 new submission_id and model information per model from `image_metadata` 
    table that doesn't exist in `predictions`
    """
    query = f"""
    WITH ranked_submissions AS (
        SELECT 
            m.submission_id,
            m.model,
            m.url,
            m.date,
            ROW_NUMBER() OVER (PARTITION BY m.model ORDER BY m.date DESC) as rn
        FROM 
            `{PROJECT_ID}.{DATASET_ID}.{METADATA_TABLE}` m
        LEFT JOIN 
            `{PROJECT_ID}.{DATASET_ID}.{PREDICTIONS_TABLE}` p
        ON 
            m.submission_id = p.submission_id
        WHERE 
            p.submission_id IS NULL
    )
    SELECT 
        submission_id,
        model,
        url as image_url,
        date as created_at
    FROM 
        ranked_submissions
    WHERE 
        rn <= 5
    ORDER BY 
        model, date DESC
    """
    
    try:
        df = client.query(query).to_dataframe()
        logger.info(f"Found {len(df)} new submissions to process")
        return df
    except Exception as e:
        logger.error(f"Error querying new submissions: {e}")
        raise


def download_image(image_url, submission_id):
    """
    Download image from URL and save to prediction/images folder
    
    Returns:
        Path to downloaded image or None if failed
    """
    try:
        logger.info(f"Downloading image for {submission_id} from {image_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        
        file_extension = image_url.split('.')[-1].split('?')[0]
        if file_extension.lower() not in ['jpg', 'jpeg', 'png', 'webp']:
            file_extension = 'jpg'
        
        image_path = IMAGE_DIR / f"{submission_id}.{file_extension}"
        img.save(image_path)
        
        logger.info(f"Saved image to {image_path}")
        return image_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {image_url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing image for {submission_id}: {e}")
        return None


def check_susy_correct(susy_prediction, true_model):
    """
    Check if SuSy prediction matches the true model
    
    Maps model names to handle different naming conventions:
    - "dalle2" or "dalle3" -> matches SuSy "dalle3"
    - "midjourney" -> matches SuSy "midjourney"
    - "Art" or "authentic" -> matches SuSy "authentic"
    """
    # Normalize names
    susy_pred = susy_prediction.lower()
    true_mod = true_model.lower()
    
    # Direct matches
    if susy_pred == true_mod:
        return True
    
    # Handle DALL-E variations
    if susy_pred == "dalle3" and true_mod in ["dalle2", "dalle3", "dalle"]:
        return True
    
    # Handle MidJourney variations
    if susy_pred == "midjourney" and true_mod in ["midjourney", "mj"]:
        return True
    
    # Handle authentic/art variations
    if susy_pred == "authentic" and true_mod in ["art", "authentic", "real"]:
        return True
    
    return False


def get_prediction(submission_id, model, image_url):
    """
    Download image, run the pipeline, and return prediction results
    """
    image_path = download_image(image_url, submission_id)
    
    if image_path is None:
        raise Exception(f"Failed to download image for {submission_id}")
    
    try:
        logger.info(f"Running pipeline on {image_path}")
        result = pipeline.analyze(str(image_path), topk_styles=5)
        
        clip_styles = result["styles"]
        blip_caption = result["caption"]
        susy_result = result["susy"]
        
        prediction = {
            "submission_id": submission_id,
            "model": model,
            "image_url": image_url,
            "BLIP_result": blip_caption,
            "created_at": datetime.now()
        }
                
        # Handle if styles is a tuple of (style_name, score) pairs
        if isinstance(clip_styles, (list, tuple)) and len(clip_styles) > 0:
            # Check if it's a list of tuples or list of dicts
            if isinstance(clip_styles[0], tuple):
                # Format: [(style_name, score), ...]
                for i in range(5):
                    if i < len(clip_styles):
                        prediction[f"CLIP_top_{i+1}_mod"] = clip_styles[i][0]
                        prediction[f"CLIP_top_{i+1}_prob"] = float(clip_styles[i][1])
                    else:
                        prediction[f"CLIP_top_{i+1}_mod"] = None
                        prediction[f"CLIP_top_{i+1}_prob"] = None
            elif isinstance(clip_styles[0], dict):
                # Format: [{"style": ..., "score": ...}, ...]
                for i in range(5):
                    if i < len(clip_styles):
                        prediction[f"CLIP_top_{i+1}_mod"] = clip_styles[i]["style"]
                        prediction[f"CLIP_top_{i+1}_prob"] = float(clip_styles[i]["score"])
                    else:
                        prediction[f"CLIP_top_{i+1}_mod"] = None
                        prediction[f"CLIP_top_{i+1}_prob"] = None
            else:
                raise ValueError(f"Unexpected clip_styles format: {type(clip_styles[0])}")
        else:
            # No styles returned
            for i in range(5):
                prediction[f"CLIP_top_{i+1}_mod"] = None
                prediction[f"CLIP_top_{i+1}_prob"] = None
        
        prediction["SuSy_model"] = susy_result["pred_class"]
        
        # Check if SuSy prediction is correct
        prediction["SuSy_correct"] = check_susy_correct(susy_result["pred_class"], model)
        
        if susy_result["model_type"] == "transfer_learning_3class":
            prediction["SuSy_dalle_prob"] = float(susy_result["probs"].get("dalle3", 0.0))
            prediction["SuSy_midjourney_prob"] = float(susy_result["probs"].get("midjourney", 0.0))
            prediction["SuSy_authentic_prob"] = float(susy_result["probs"].get("authentic", 0.0))
        else:
            prediction["SuSy_dalle_prob"] = float(susy_result["probs"].get("dalle-3-images", 0.0))
            prediction["SuSy_midjourney_prob"] = float(
                susy_result["probs"].get("midjourney-images", 0.0) + 
                susy_result["probs"].get("midjourney_tti", 0.0)
            )
            prediction["SuSy_authentic_prob"] = float(susy_result["probs"].get("authentic", 0.0))
        
        logger.info(f"Successfully processed {submission_id}: SuSy={prediction['SuSy_model']}, True={model}, Correct={prediction['SuSy_correct']}")
        return prediction
        
    except Exception as e:
        logger.error(f"Error getting prediction for {submission_id}: {e}")
        raise


def save_predictions(predictions):
    """
    Save prediction results to BigQuery and CSV
    """
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{PREDICTIONS_TABLE}"
    
    try:
        df = pd.DataFrame(predictions)
        
        # Save to BigQuery FIRST (create a copy without image_url)
        df_bq = df.drop(columns=['image_url']).copy()
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
        )
        
        job = client.load_table_from_dataframe(df_bq, table_id, job_config=job_config)
        job.result()
        
        logger.info(f"Saved {len(predictions)} predictions to {table_id}")
        
        # Save to CSV with image_url (use original df)
        csv_filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = RESULTS_DIR / csv_filename
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to CSV: {csv_path}")
        
        # Also append to a master CSV file
        master_csv_path = RESULTS_DIR / "all_predictions.csv"
        if master_csv_path.exists():
            df.to_csv(master_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(master_csv_path, index=False)
        logger.info(f"Appended predictions to master CSV: {master_csv_path}")
        
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise


def run_pipeline():
    """
    Main function to run the entire pipeline
    """
    logger.info("Starting prediction pipeline")
    
    create_pred_table()
    
    new_submissions = get_new_submissions()
    
    if len(new_submissions) == 0:
        logger.info("No new submissions to process")
        return
    
    predictions = []
    
    for _, row in new_submissions.iterrows():
        try:
            logger.info(f"Processing submission {row['submission_id']} (model: {row['model']})")
            
            prediction = get_prediction(
                submission_id=row['submission_id'],
                model=row['model'],
                image_url=row['image_url']
            )
            
            predictions.append(prediction)
            
        except Exception as e:
            logger.error(f"Failed to process submission {row['submission_id']}: {e}")
            continue
    
    if predictions:
        save_predictions(predictions)
        logger.info(f"Pipeline completed. Processed {len(predictions)} submissions")
    else:
        logger.warning("No predictions were generated")


if __name__ == "__main__":
    run_pipeline()