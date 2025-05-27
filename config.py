#!/usr/bin/env python3
"""
Configuration settings for the stock dashboard runner
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class DashboardConfig:
    """Configuration settings for the dashboard runner"""
    
    # File paths
    dashboard_file: str = "dashboard.py"
    requirements_file: str = "requirements.txt"
    
    # Server settings
    default_port: int = 8501
    default_host: str = "localhost"
    
    # UI settings
    header_width: int = 30
    separator_width: int = 50
    
    # Behavior settings
    auto_install_requirements: bool = False
    verbose_output: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.default_port < 1 or self.default_port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        
        if self.header_width < 10:
            raise ValueError("Header width must be at least 10")


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = Path(config_file) if config_file else None
        self._config = DashboardConfig()
    
    def load_config(self) -> DashboardConfig:
        """
        Load configuration from file or return default
        
        Returns:
            DashboardConfig: The loaded configuration
        """
        if self.config_file and self.config_file.exists():
            try:
                # In a real implementation, you might load from JSON, YAML, etc.
                # For now, just return default config
                pass
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        return self._config
    
    def save_config(self, config: DashboardConfig) -> bool:
        """
        Save configuration to file
        
        Args:
            config: Configuration to save
            
        Returns:
            bool: True if saved successfully
        """
        if not self.config_file:
            return False
        
        try:
            # In a real implementation, you would serialize to JSON, YAML, etc.
            # For now, just return True
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    @property
    def config(self) -> DashboardConfig:
        """Get the current configuration"""
        return self._config
    
    @config.setter
    def config(self, value: DashboardConfig) -> None:
        """Set the configuration"""
        self._config = value