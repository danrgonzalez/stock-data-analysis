#!/usr/bin/env python3
"""
Main application runner for the stock dashboard
"""
import sys
from pathlib import Path
from typing import Optional

# Import our custom modules
from config import ConfigManager, DashboardConfig
from package_manager import PackageManager
from dashboard_runner import DashboardRunner
from user_interface import UserInterface


class StockDashboardApp:
    """Main application class that coordinates all modules"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.load_config()
        self.package_manager = PackageManager(self.config.requirements_file)
        self.dashboard_runner = DashboardRunner(self.config.dashboard_file)
        self.ui = UserInterface()
    
    def validate_environment(self) -> bool:
        """
        Validate that all required files exist and environment is ready
        
        Returns:
            bool: True if environment is valid
        """
        # Check if dashboard file exists
        if not self.dashboard_runner.check_dashboard_exists():
            self.ui.print_error(f"Dashboard file '{self.config.dashboard_file}' not found in current directory")
            self.ui.print_info("Please make sure you're in the correct directory")
            return False
        
        # Check if requirements file exists (optional)
        if not self.package_manager.check_requirements_exist():
            self.ui.print_warning(f"Requirements file '{self.config.requirements_file}' not found")
            self.ui.print_info("Skipping package installation step")
        
        return True
    
    def handle_package_installation(self) -> bool:
        """
        Handle package installation based on user choice or config
        
        Returns:
            bool: True if packages are ready (installed or skipped)
        """
        if not self.package_manager.check_requirements_exist():
            return True  # No requirements file, nothing to install
        
        # Check if we should auto-install or ask user
        if self.config.auto_install_requirements:
            should_install = True
        else:
            should_install = self.ui.get_yes_no_input(
                "Install/update requirements?", 
                default=True
            )
        
        if should_install:
            return self.package_manager.install_requirements(
                verbose=self.config.verbose_output
            )
        
        return True
    
    def run_dashboard_with_options(self) -> None:
        """Run the dashboard with optional configuration"""
        try:
            self.dashboard_runner.run_streamlit_dashboard(
                port=self.config.default_port,
                host=self.config.default_host
            )
        except FileNotFoundError as e:
            self.ui.print_error(str(e))
            sys.exit(1)
        except Exception as e:
            self.ui.print_error(f"Unexpected error: {e}")
            sys.exit(1)
    
    def run(self) -> None:
        """Main application entry point"""
        # Print header
        self.ui.print_header("🏃‍♂️ Stock Dashboard Runner", self.config.header_width)
        
        # Validate environment
        if not self.validate_environment():
            sys.exit(1)
        
        # Handle package installation
        if not self.handle_package_installation():
            self.ui.print_error("Failed to install required packages")
            sys.exit(1)
        
        # Run the dashboard
        self.run_dashboard_with_options()


def main():
    """Entry point for the application"""
    try:
        app = StockDashboardApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()