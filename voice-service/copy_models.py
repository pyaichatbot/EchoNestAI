import os
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Source and destination directories
TTS_CACHE_DIR = os.path.expanduser("~/Library/Application Support/tts")
LOCAL_MODELS_DIR = "models/coqui"

def copy_models():
    """Copy TTS models from system cache to local directory."""
    try:
        # Create local directory if it doesn't exist
        os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
        
        # List of models to copy
        models = [
            "tts_models--en--ljspeech--tacotron2-DDC",
            "tts_models--en--vctk--vits",
            "tts_models--de--thorsten--tacotron2-DCA",
            "vocoder_models--en--ljspeech--hifigan_v2",
            "vocoder_models--de--thorsten--fullband-melgan"
        ]
        
        for model in models:
            src_path = os.path.join(TTS_CACHE_DIR, model)
            dst_path = os.path.join(LOCAL_MODELS_DIR, model)
            
            if os.path.exists(src_path):
                if os.path.exists(dst_path):
                    logger.info(f"Model {model} already exists in local directory, skipping...")
                    continue
                    
                logger.info(f"Copying {model} to local directory...")
                shutil.copytree(src_path, dst_path)
                logger.info(f"Successfully copied {model}")
            else:
                logger.warning(f"Model {model} not found in system cache")
                
        logger.info("Model copying completed!")
        
    except Exception as e:
        logger.error(f"Error copying models: {e}")
        raise

if __name__ == "__main__":
    copy_models() 