#!/usr/bin/env python3
"""
Stock Dashboard Runner Package

A modular Python application for running Streamlit-based stock dashboards
with automatic dependency management and user-friendly interface.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .config import DashboardConfig, ConfigManager
from .package_manager import PackageManager
from .dashboard_runner import DashboardRunner
from .user_interface import UserInterface
from .main_runner import StockDashboardApp

__all__ = [
    "DashboardConfig",
    "ConfigManager", 
    "PackageManager",
    "DashboardRunner",
    "UserInterface",
    "StockDashboardApp"
]