import os
import json
import logging
import subprocess
import time
import signal
import psutil
from pathlib import Path

logger = logging.getLogger("echonest_device.offline_server_manager")

class OfflineServerManager:
    """
    Manages the offline backend server for the EchoNest AI device.
    Handles starting, stopping, and monitoring the server.
    """
    
    def __init__(self, server_url, models_path=None, content_path=None, cache_path=None):
        """
        Initialize the offline server manager.
        
        Args:
            server_url: URL of the offline server
            models_path: Path to models directory
            content_path: Path to content directory
            cache_path: Path to cache directory
        """
        self.server_url = server_url
        self.models_path = models_path
        self.content_path = content_path
        self.cache_path = cache_path
        self.server_process = None
        self.pid_file = Path(os.path.expanduser("~")) / ".echonest" / "offline_server.pid"
        
        # Create .echonest directory if it doesn't exist
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
    
    def start_server(self):
        """
        Start the offline backend server.
        
        Returns:
            bool: True if server started successfully
        """
        if self.is_server_running():
            logger.info("Offline server is already running")
            return True
        
        try:
            # Prepare environment variables
            env = os.environ.copy()
            
            if self.models_path:
                env["ECHONEST_MODELS_PATH"] = str(self.models_path)
            
            if self.content_path:
                env["ECHONEST_CONTENT_PATH"] = str(self.content_path)
            
            if self.cache_path:
                env["ECHONEST_CACHE_PATH"] = str(self.cache_path)
            
            # Get the path to the offline server script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            server_script = os.path.join(script_dir, "..", "offline_server.py")
            
            # Start the server process
            self.server_process = subprocess.Popen(
                ["python", server_script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Detach from parent process
            )
            
            # Save PID to file
            with open(self.pid_file, 'w') as f:
                f.write(str(self.server_process.pid))
            
            # Wait for server to start
            max_retries = 10
            retries = 0
            
            while retries < max_retries:
                if self._check_server_health():
                    logger.info(f"Offline server started successfully with PID {self.server_process.pid}")
                    return True
                
                time.sleep(1)
                retries += 1
            
            # If we get here, server didn't start properly
            logger.error("Offline server failed to start within timeout")
            self.stop_server()
            return False
        except Exception as e:
            logger.error(f"Error starting offline server: {e}")
            return False
    
    def stop_server(self):
        """
        Stop the offline backend server.
        
        Returns:
            bool: True if server stopped successfully
        """
        try:
            pid = self._get_server_pid()
            
            if pid:
                # Try to terminate gracefully first
                try:
                    process = psutil.Process(pid)
                    process.terminate()
                    
                    # Wait for process to terminate
                    gone, alive = psutil.wait_procs([process], timeout=5)
                    
                    # If still alive, kill forcefully
                    if alive:
                        process.kill()
                except psutil.NoSuchProcess:
                    # Process already gone
                    pass
                
                # Remove PID file
                if self.pid_file.exists():
                    self.pid_file.unlink()
                
                logger.info(f"Offline server with PID {pid} stopped successfully")
                return True
            else:
                logger.info("No running offline server found")
                return True
        except Exception as e:
            logger.error(f"Error stopping offline server: {e}")
            return False
    
    def is_server_running(self):
        """
        Check if the offline server is running.
        
        Returns:
            bool: True if server is running
        """
        pid = self._get_server_pid()
        
        if pid:
            try:
                # Check if process exists
                process = psutil.Process(pid)
                
                # Check if it's a Python process (additional validation)
                if "python" in process.name().lower():
                    # Check server health
                    return self._check_server_health()
            except psutil.NoSuchProcess:
                # Process doesn't exist, clean up PID file
                if self.pid_file.exists():
                    self.pid_file.unlink()
        
        return False
    
    def _get_server_pid(self):
        """
        Get the PID of the running server from the PID file.
        
        Returns:
            int: Server PID, or None if not found
        """
        try:
            if self.pid_file.exists():
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                return pid
            return None
        except Exception as e:
            logger.error(f"Error reading server PID file: {e}")
            return None
    
    def _check_server_health(self):
        """
        Check if the server is healthy by making a request to the health endpoint.
        
        Returns:
            bool: True if server is healthy
        """
        import requests
        
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_server_status(self):
        """
        Get detailed status of the offline server.
        
        Returns:
            dict: Server status information
        """
        status = {
            "running": self.is_server_running(),
            "pid": self._get_server_pid(),
            "url": self.server_url
        }
        
        # If server is running, get additional information
        if status["running"]:
            try:
                import requests
                
                # Get server metrics
                response = requests.get(f"{self.server_url}/metrics", timeout=2)
                if response.status_code == 200:
                    status["metrics"] = response.json()
                
                # Get server version
                response = requests.get(f"{self.server_url}/version", timeout=2)
                if response.status_code == 200:
                    status["version"] = response.json().get("version")
            except requests.exceptions.RequestException:
                pass
        
        return status
