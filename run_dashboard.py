#!/usr/bin/env python3
"""
Simple script to run the stock dashboard
"""
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("Starting the dashboard...")
    print("🚀 Dashboard will open in your default browser")
    print("📊 Upload your StockData.xlsx file when prompted")
    print("Press Ctrl+C to stop the dashboard\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    print("🏃‍♂️ Stock Dashboard Runner")
    print("=" * 30)
    
    # Check if dashboard.py exists
    if not Path("dashboard.py").exists():
        print("❌ dashboard.py not found in current directory")
        print("Please make sure you're in the correct directory")
        return
    
    # Install requirements if needed
    choice = input("Install/update requirements? (y/n, default=y): ").lower()
    if choice != 'n':
        if not install_requirements():
            return
    
    # Run dashboard
    run_dashboard()

if __name__ == "__main__":
    main()
