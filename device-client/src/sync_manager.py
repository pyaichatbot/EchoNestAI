import os
import json
import logging
import time
import threading
import subprocess
from pathlib import Path

logger = logging.getLogger("echonest_device.sync_manager")

class SyncManager:
    """
    Manages content synchronization between the device and the backend.
    Handles downloading content, models, and tracking sync status.
    """
    
    def __init__(self, device_id, device_token, api_url, content_path, models_path, cache_path):
        """
        Initialize the sync manager.
        
        Args:
            device_id: Device ID
            device_token: Device authentication token
            api_url: Backend API URL
            content_path: Path to store content
            models_path: Path to store models
            cache_path: Path to store cache
        """
        self.device_id = device_id
        self.device_token = device_token
        self.api_url = api_url
        self.content_path = Path(content_path)
        self.models_path = Path(models_path)
        self.cache_path = Path(cache_path)
        
        # Create necessary directories
        self.content_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize API client
        from .api_client import APIClient
        self.api_client = APIClient(
            api_url=api_url,
            device_id=device_id,
            device_token=device_token
        )
        
        # Load sync state
        self.sync_state_path = self.cache_path / 'sync_state.json'
        self.sync_state = self._load_sync_state()
        
        # Initialize sync lock
        self.sync_lock = threading.Lock()
        self.current_sync_id = None
        self.sync_in_progress = False
    
    def _load_sync_state(self):
        """
        Load synchronization state from file.
        
        Returns:
            dict: Synchronization state
        """
        try:
            if self.sync_state_path.exists():
                with open(self.sync_state_path, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "last_sync": None,
                    "content_items": {},
                    "models": {},
                    "sync_history": []
                }
        except Exception as e:
            logger.error(f"Error loading sync state: {e}")
            return {
                "last_sync": None,
                "content_items": {},
                "models": {},
                "sync_history": []
            }
    
    def _save_sync_state(self):
        """
        Save synchronization state to file.
        """
        try:
            with open(self.sync_state_path, 'w') as f:
                json.dump(self.sync_state, f, indent=2)
            logger.info(f"Sync state saved to {self.sync_state_path}")
        except Exception as e:
            logger.error(f"Error saving sync state: {e}")
    
    def sync_content(self, force=False):
        """
        Synchronize content with the backend.
        
        Args:
            force: Force full synchronization
            
        Returns:
            dict: Synchronization result
        """
        # Ensure only one sync operation runs at a time
        if not self.sync_lock.acquire(blocking=False):
            logger.warning("Sync already in progress")
            return {"success": False, "message": "Sync already in progress"}
        
        try:
            self.sync_in_progress = True
            self.current_sync_id = f"sync_{int(time.time())}"
            
            # Get sync manifest from backend
            manifest = self.api_client.get_sync_manifest()
            
            if not manifest.get('success'):
                logger.error(f"Failed to get sync manifest: {manifest.get('message')}")
                return {"success": False, "message": manifest.get('message')}
            
            # Extract manifest data
            sync_id = manifest.get('sync_id', self.current_sync_id)
            self.current_sync_id = sync_id
            content_items = manifest.get('content_items', [])
            models = manifest.get('models', [])
            
            # Update sync status to in_progress
            self.api_client.update_sync_status(
                sync_id=sync_id,
                status="in_progress",
                progress=0,
                details="Starting content synchronization"
            )
            
            # Calculate total items to sync
            total_items = len(content_items) + len(models)
            synced_items = 0
            failed_items = 0
            skipped_items = 0
            
            # Sync content items
            for item in content_items:
                item_id = item.get('id')
                item_hash = item.get('hash')
                item_path = item.get('path')
                item_type = item.get('type', 'document')
                
                # Determine local path
                local_path = self.content_path / item_path
                
                # Check if item exists and hash matches (unless force sync)
                existing_item = self.sync_state['content_items'].get(item_id, {})
                existing_hash = existing_item.get('hash')
                
                if not force and existing_hash == item_hash and local_path.exists():
                    logger.info(f"Skipping content item {item_id}: already up to date")
                    skipped_items += 1
                else:
                    # Download content item
                    success = self.api_client.download_content(item_id, str(local_path))
                    
                    if success:
                        # Update sync state
                        self.sync_state['content_items'][item_id] = {
                            'hash': item_hash,
                            'path': item_path,
                            'type': item_type,
                            'synced_at': time.time()
                        }
                        synced_items += 1
                    else:
                        failed_items += 1
                
                # Update progress
                progress = int((synced_items + skipped_items + failed_items) / total_items * 100)
                self.api_client.update_sync_status(
                    sync_id=sync_id,
                    status="in_progress",
                    progress=progress,
                    details=f"Synced {synced_items} items, skipped {skipped_items}, failed {failed_items}"
                )
            
            # Sync models
            for model in models:
                model_id = model.get('id')
                model_hash = model.get('hash')
                model_path = model.get('path')
                model_type = model.get('type', 'embedding')
                
                # Determine local path
                local_path = self.models_path / model_path
                
                # Check if model exists and hash matches (unless force sync)
                existing_model = self.sync_state['models'].get(model_id, {})
                existing_hash = existing_model.get('hash')
                
                if not force and existing_hash == model_hash and local_path.exists():
                    logger.info(f"Skipping model {model_id}: already up to date")
                    skipped_items += 1
                else:
                    # Download model
                    success = self.api_client.download_model(model_id, str(local_path))
                    
                    if success:
                        # Update sync state
                        self.sync_state['models'][model_id] = {
                            'hash': model_hash,
                            'path': model_path,
                            'type': model_type,
                            'synced_at': time.time()
                        }
                        synced_items += 1
                    else:
                        failed_items += 1
                
                # Update progress
                progress = int((synced_items + skipped_items + failed_items) / total_items * 100)
                self.api_client.update_sync_status(
                    sync_id=sync_id,
                    status="in_progress",
                    progress=progress,
                    details=f"Synced {synced_items} items, skipped {skipped_items}, failed {failed_items}"
                )
            
            # Update sync history
            self.sync_state['last_sync'] = time.time()
            self.sync_state['sync_history'].append({
                'sync_id': sync_id,
                'timestamp': time.time(),
                'items_synced': synced_items,
                'items_skipped': skipped_items,
                'items_failed': failed_items
            })
            
            # Limit sync history to last 10 entries
            if len(self.sync_state['sync_history']) > 10:
                self.sync_state['sync_history'] = self.sync_state['sync_history'][-10:]
            
            # Save sync state
            self._save_sync_state()
            
            # Update sync status to completed
            status = "completed" if failed_items == 0 else "completed_with_errors"
            self.api_client.update_sync_status(
                sync_id=sync_id,
                status=status,
                progress=100,
                details=f"Sync completed. Synced {synced_items} items, skipped {skipped_items}, failed {failed_items}"
            )
            
            return {
                "success": True,
                "sync_id": sync_id,
                "items_synced": synced_items,
                "items_skipped": skipped_items,
                "items_failed": failed_items
            }
        except Exception as e:
            logger.error(f"Error during content synchronization: {e}")
            
            # Update sync status to failed
            if self.current_sync_id:
                self.api_client.update_sync_status(
                    sync_id=self.current_sync_id,
                    status="failed",
                    details=f"Sync failed: {str(e)}"
                )
            
            return {"success": False, "message": str(e)}
        finally:
            self.sync_in_progress = False
            self.current_sync_id = None
            self.sync_lock.release()
    
    def get_content_item_path(self, content_id):
        """
        Get the local path for a content item.
        
        Args:
            content_id: Content item ID
            
        Returns:
            Path: Local path to the content item, or None if not found
        """
        item = self.sync_state['content_items'].get(content_id)
        
        if item and 'path' in item:
            local_path = self.content_path / item['path']
            if local_path.exists():
                return local_path
        
        return None
    
    def get_model_path(self, model_id):
        """
        Get the local path for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Path: Local path to the model, or None if not found
        """
        model = self.sync_state['models'].get(model_id)
        
        if model and 'path' in model:
            local_path = self.models_path / model['path']
            if local_path.exists():
                return local_path
        
        return None
    
    def get_sync_status(self):
        """
        Get the current synchronization status.
        
        Returns:
            dict: Synchronization status
        """
        return {
            "in_progress": self.sync_in_progress,
            "current_sync_id": self.current_sync_id,
            "last_sync": self.sync_state.get('last_sync'),
            "content_items_count": len(self.sync_state.get('content_items', {})),
            "models_count": len(self.sync_state.get('models', {})),
            "sync_history": self.sync_state.get('sync_history', [])
        }
    
    def get_content_items(self, content_type=None):
        """
        Get a list of synced content items.
        
        Args:
            content_type: Optional filter by content type
            
        Returns:
            list: Content items
        """
        items = []
        
        for item_id, item_data in self.sync_state['content_items'].items():
            if content_type is None or item_data.get('type') == content_type:
                items.append({
                    'id': item_id,
                    'path': item_data.get('path'),
                    'type': item_data.get('type'),
                    'synced_at': item_data.get('synced_at')
                })
        
        return items
    
    def get_models(self, model_type=None):
        """
        Get a list of synced models.
        
        Args:
            model_type: Optional filter by model type
            
        Returns:
            list: Models
        """
        models = []
        
        for model_id, model_data in self.sync_state['models'].items():
            if model_type is None or model_data.get('type') == model_type:
                models.append({
                    'id': model_id,
                    'path': model_data.get('path'),
                    'type': model_data.get('type'),
                    'synced_at': model_data.get('synced_at')
                })
        
        return models
