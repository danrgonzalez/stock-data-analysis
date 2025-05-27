#!/usr/bin/env python3
"""
Dashboard execution utilities for the stock dashboard
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional, List


class DashboardRunner:
    """Handles running and managing the Streamlit dashboard"""
    
    def __init__(self, dashboard_file: str = "dashboard.py"):
        self.dashboard_file = Path(dashboard_file)
    
    def check_dashboard_exists(self) -> bool:
        """Check if dashboard file exists"""
        return self.dashboard_file.exists()
    
    def run_streamlit_dashboard(self, 
                              port: Optional[int] = None, 
                              host: str = "localhost",
                              additional_args: Optional[List[str]] = None) -> None:
        """
        Run the Streamlit dashboard
        
        Args:
            port: Port number to run on (optional)
            host: Host address to bind to
            additional_args: Additional arguments to pass to streamlit
        """
        if not self.check_dashboard_exists():
            raise FileNotFoundError(f"Dashboard file '{self.dashboard_file}' not found")
        
        # Build command
        cmd = [sys.executable, "-m", "streamlit", "run", str(self.dashboard_file)]
        
        if port:
            cmd.extend(["--server.port", str(port)])
        
        if host != "localhost":
            cmd.extend(["--server.address", host])
        
        if additional_args:
            cmd.extend(additional_args)
        
        self._print_startup_messages()
        
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running dashboard: {e}")
            raise
    
    def _print_startup_messages(self) -> None:
        """Print informative startup messages"""
        print("Starting the dashboard...")
        print("🚀 Dashboard will open in your default browser")
        print("📊 Upload your StockData.xlsx file when prompted")
        print("Press Ctrl+C to stop the dashboard\n")
    
    def get_dashboard_url(self, port: int = 8501, host: str = "localhost") -> str:
        """
        Get the URL where the dashboard will be accessible
        
        Args:
            port: Port number (default Streamlit port is 8501)
            host: Host address
            
        Returns:
            str: Full URL to access the dashboard
        """
        return f"http://{host}:{port}"