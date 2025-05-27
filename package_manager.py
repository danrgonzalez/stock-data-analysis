#!/usr/bin/env python3
"""
Package management utilities for the stock dashboard
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional


class PackageManager:
    """Handles package installation and dependency management"""
    
    def __init__(self, requirements_file: str = "requirements.txt"):
        self.requirements_file = Path(requirements_file)
    
    def check_requirements_exist(self) -> bool:
        """Check if requirements file exists"""
        return self.requirements_file.exists()
    
    def install_requirements(self, verbose: bool = True) -> bool:
        """
        Install required packages from requirements.txt
        
        Args:
            verbose: Whether to print status messages
            
        Returns:
            bool: True if installation successful, False otherwise
        """
        if not self.check_requirements_exist():
            if verbose:
                print(f"❌ Requirements file '{self.requirements_file}' not found")
            return False
        
        if verbose:
            print(f"Installing packages from {self.requirements_file}...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)],
                capture_output=not verbose,
                text=True,
                check=True
            )
            
            if verbose:
                print("✅ Requirements installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"❌ Error installing requirements: {e}")
                if hasattr(e, 'stderr') and e.stderr:
                    print(f"Error details: {e.stderr}")
            return False
    
    def check_package_installed(self, package_name: str) -> bool:
        """
        Check if a specific package is installed
        
        Args:
            package_name: Name of the package to check
            
        Returns:
            bool: True if package is installed, False otherwise
        """
        try:
            subprocess.run(
                [sys.executable, "-c", f"import {package_name}"],
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False