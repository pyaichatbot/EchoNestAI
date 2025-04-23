import os
import json
import logging
import requests
from typing import Dict, Any, Optional, List

logger = logging.getLogger("echonest_device.api_client")

class APIClient:
    """
    Client for communicating with the EchoNest AI backend API.
    Handles authentication, device registration, and API requests.
    """
    
    def __init__(self, api_url: str, device_id: Optional[str] = None, device_token: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_url: Backend API URL
            device_id: Optional device ID for authenticated requests
            device_token: Optional device token for authenticated requests
        """
        self.api_url = api_url
        self.device_id = device_id
        self.device_token = device_token
        
        # Set up session for connection pooling
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "User-Agent": "EchoNest-Device-Client/1.0",
            "Content-Type": "application/json"
        })
        
        logger.info(f"API client initialized with URL: {api_url}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Returns:
            dict: Authentication headers
        """
        headers = {}
        
        if self.device_id and self.device_token:
            headers["X-Device-ID"] = self.device_id
            headers["X-Device-Token"] = self.device_token
        
        return headers
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, 
                     params: Optional[Dict[str, Any]] = None, auth_required: bool = True,
                     files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an API request.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Optional request data
            params: Optional query parameters
            auth_required: Whether authentication is required
            files: Optional files to upload
            
        Returns:
            dict: API response
        """
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        headers = {}
        
        if auth_required:
            if not self.device_id or not self.device_token:
                logger.error("Authentication required but device credentials not provided")
                return {"success": False, "message": "Authentication required"}
            
            headers.update(self._get_auth_headers())
        
        try:
            if files:
                # For multipart/form-data requests (file uploads)
                response = self.session.request(
                    method=method,
                    url=url,
                    data=data,
                    params=params,
                    headers=headers,
                    files=files,
                    timeout=30
                )
            else:
                # For JSON requests
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=headers,
                    timeout=30
                )
            
            # Parse response
            if response.status_code == 204:
                return {"success": True}
            
            try:
                result = response.json()
            except ValueError:
                result = {"success": False, "message": "Invalid JSON response"}
            
            # Add status code to result
            result["status_code"] = response.status_code
            
            # Check for success
            if 200 <= response.status_code < 300:
                result["success"] = True
            else:
                result["success"] = False
                logger.error(f"API request failed: {response.status_code} - {result.get('message', 'Unknown error')}")
            
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            return {"success": False, "message": str(e)}
    
    def register_device(self, registration_code: str, device_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Register the device with the backend.
        
        Args:
            registration_code: Registration code from parent dashboard
            device_name: Optional device name
            
        Returns:
            dict: Registration result
        """
        data = {
            "registration_code": registration_code
        }
        
        if device_name:
            data["device_name"] = device_name
        
        return self._make_request(
            method="POST",
            endpoint="/api/devices/register",
            data=data,
            auth_required=False
        )
    
    def get_sync_manifest(self) -> Dict[str, Any]:
        """
        Get the content synchronization manifest.
        
        Returns:
            dict: Sync manifest
        """
        return self._make_request(
            method="GET",
            endpoint="/api/devices/sync/manifest",
            auth_required=True
        )
    
    def update_sync_status(self, sync_id: str, status: str, progress: int = 0, details: Optional[str] = None) -> Dict[str, Any]:
        """
        Update synchronization status.
        
        Args:
            sync_id: Sync operation ID
            status: Status (in_progress, completed, failed, completed_with_errors)
            progress: Progress percentage (0-100)
            details: Optional status details
            
        Returns:
            dict: Update result
        """
        data = {
            "sync_id": sync_id,
            "status": status,
            "progress": progress
        }
        
        if details:
            data["details"] = details
        
        return self._make_request(
            method="POST",
            endpoint="/api/devices/sync/status",
            data=data,
            auth_required=True
        )
    
    def download_content(self, content_id: str, destination_path: str) -> bool:
        """
        Download content item.
        
        Args:
            content_id: Content item ID
            destination_path: Path to save content
            
        Returns:
            bool: True if download was successful
        """
        try:
            url = f"{self.api_url}/api/devices/content/{content_id}/download"
            headers = self._get_auth_headers()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Stream download to file
            with self.session.get(url, headers=headers, stream=True, timeout=60) as response:
                if response.status_code != 200:
                    logger.error(f"Content download failed: {response.status_code}")
                    return False
                
                with open(destination_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"Content {content_id} downloaded to {destination_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading content {content_id}: {e}")
            return False
    
    def download_model(self, model_id: str, destination_path: str) -> bool:
        """
        Download model.
        
        Args:
            model_id: Model ID
            destination_path: Path to save model
            
        Returns:
            bool: True if download was successful
        """
        try:
            url = f"{self.api_url}/api/devices/models/{model_id}/download"
            headers = self._get_auth_headers()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Stream download to file
            with self.session.get(url, headers=headers, stream=True, timeout=120) as response:
                if response.status_code != 200:
                    logger.error(f"Model download failed: {response.status_code}")
                    return False
                
                with open(destination_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"Model {model_id} downloaded to {destination_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            return False
    
    def get_language_resources(self, language: str) -> Dict[str, Any]:
        """
        Get language-specific resources.
        
        Args:
            language: Language code
            
        Returns:
            dict: Language resources
        """
        return self._make_request(
            method="GET",
            endpoint=f"/api/languages/{language}/resources",
            auth_required=True
        )
    
    def report_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Report device metrics to backend.
        
        Args:
            metrics: Metrics data
            
        Returns:
            dict: Report result
        """
        return self._make_request(
            method="POST",
            endpoint="/api/devices/metrics",
            data=metrics,
            auth_required=True
        )
    
    def report_error(self, error_type: str, error_message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Report device error to backend.
        
        Args:
            error_type: Error type
            error_message: Error message
            details: Optional error details
            
        Returns:
            dict: Report result
        """
        data = {
            "error_type": error_type,
            "error_message": error_message
        }
        
        if details:
            data["details"] = details
        
        return self._make_request(
            method="POST",
            endpoint="/api/devices/errors",
            data=data,
            auth_required=True
        )
    
    def check_for_updates(self) -> Dict[str, Any]:
        """
        Check for device software updates.
        
        Returns:
            dict: Update information
        """
        return self._make_request(
            method="GET",
            endpoint="/api/devices/updates",
            auth_required=True
        )
    
    def download_update(self, update_id: str, destination_path: str) -> bool:
        """
        Download device software update.
        
        Args:
            update_id: Update ID
            destination_path: Path to save update
            
        Returns:
            bool: True if download was successful
        """
        try:
            url = f"{self.api_url}/api/devices/updates/{update_id}/download"
            headers = self._get_auth_headers()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Stream download to file
            with self.session.get(url, headers=headers, stream=True, timeout=120) as response:
                if response.status_code != 200:
                    logger.error(f"Update download failed: {response.status_code}")
                    return False
                
                with open(destination_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"Update {update_id} downloaded to {destination_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading update {update_id}: {e}")
            return False
