import os
import sys
import whisper
import logging
import shutil
from TTS.api import TTS
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
TTS_MODEL_CACHE_DIRS = [
    "parler_models",  # Local directory
    os.path.expanduser("~/.cache/parler_models")  # System cache as fallback
]

# Set environment variables for cache directories
os.environ["TORCH_HOME"] = os.path.abspath("models/coqui")
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("models/huggingface")
os.environ["HF_HOME"] = os.path.abspath("models/huggingface")

# TTS cache directory
TTS_CACHE_DIR = os.path.expanduser("~/Library/Application Support/tts")
LOCAL_MODELS_DIR = "models/coqui"

def verify_model_directories():
    """Verify and create necessary model directories."""
    directories = [
        "models/whisper",
        "models/coqui",
        "parler_models",
        "models/huggingface",
        "output/audio"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Verified directory: {directory}")

def check_whisper_model_exists(model_name: str, download_root: str) -> bool:
    """Check if Whisper model exists in the specified directory."""
    model_path = os.path.join(download_root, f"{model_name}.pt")
    return os.path.exists(model_path)

def check_coqui_model_exists(model_name: str) -> bool:
    """Check if Coqui model exists in local directory."""
    # Convert model name to directory format
    model_dir = model_name.replace("/", "--")
    local_path = os.path.join(LOCAL_MODELS_DIR, model_dir)
    
    # Check if model exists in local directory
    if os.path.exists(local_path):
        logger.info(f"Model {model_name} found in local directory")
        return True
        
    # Check if model exists in TTS cache
    cache_path = os.path.join(TTS_CACHE_DIR, model_dir)
    if os.path.exists(cache_path):
        logger.info(f"Model {model_name} found in TTS cache")
        return True
        
    return False

def copy_model_from_cache(model_name: str):
    """Copy model from TTS cache to local directory."""
    model_dir = model_name.replace("/", "--")
    src_path = os.path.join(TTS_CACHE_DIR, model_dir)
    dst_path = os.path.join(LOCAL_MODELS_DIR, model_dir)
    
    if os.path.exists(src_path):
        if os.path.exists(dst_path):
            logger.info(f"Model {model_name} already exists in local directory, skipping copy...")
            return
            
        logger.info(f"Copying {model_name} to local directory...")
        shutil.copytree(src_path, dst_path)
        logger.info(f"Successfully copied {model_name}")
    else:
        logger.warning(f"Model {model_name} not found in TTS cache")

def check_ai4bharat_model_exists(model_id: str, cache_dir: str) -> bool:
    """Check if AI4Bharat model exists in the cache."""
    model_path = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
    return os.path.exists(model_path)

def download_models():
    """Download all required models."""
    try:
        verify_model_directories()
        
        # Use local directory for cache
        cache_dir = "parler_models"
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")
        
        # Download Whisper models
        logger.info("Checking Whisper models...")
        for model_name in ["small", "base"]:
            if check_whisper_model_exists(model_name, "models/whisper"):
                logger.info(f"Whisper model {model_name} already exists, skipping download")
                continue
                
            try:
                logger.info(f"Downloading Whisper model: {model_name}")
                whisper.load_model(model_name, download_root="models/whisper")
                logger.info(f"Downloaded Whisper model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to download Whisper model {model_name}: {e}")
                sys.exit(1)
        
        # Download Coqui TTS models
        logger.info("Checking Coqui TTS models...")
        coqui_models = [
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/vctk/vits",
            "tts_models/de/thorsten/tacotron2-DCA"
        ]
        
        for model_name in coqui_models:
            if check_coqui_model_exists(model_name):
                # Copy from cache to local directory if needed
                copy_model_from_cache(model_name)
                continue
                
            try:
                logger.info(f"Downloading Coqui model: {model_name}")
                # Force download to local cache
                tts = TTS(model_name=model_name, progress_bar=True)
                
                # Copy from cache to local directory
                copy_model_from_cache(model_name)
                
                logger.info(f"Downloaded and copied Coqui model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to download Coqui model {model_name}: {e}")
                sys.exit(1)
        
        # Download AI4Bharat Parler-TTS model
        logger.info(f"Checking AI4Bharat Parler-TTS model: {TTS_MODEL_ID}")
        if check_ai4bharat_model_exists(TTS_MODEL_ID, cache_dir):
            logger.info("AI4Bharat model already exists, skipping download")
        else:
            try:
                # Download model
                logger.info("Downloading AI4Bharat model...")
                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    TTS_MODEL_ID,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                logger.info("Downloaded AI4Bharat model")
                
                # Download tokenizer
                logger.info("Downloading AI4Bharat tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    TTS_MODEL_ID,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                logger.info("Downloaded AI4Bharat tokenizer")
                
                # Verify model files
                model_path = os.path.join(cache_dir, "models--" + TTS_MODEL_ID.replace("/", "--"))
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model files not found in {model_path}")
                
                logger.info("All models downloaded and verified successfully!")
                
            except Exception as e:
                logger.error(f"Failed to download AI4Bharat model: {e}")
                sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during model download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_models()