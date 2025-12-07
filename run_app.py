#!/usr/bin/env python3
"""
Setup and Run Script for Streamlit Trading Strategy Application
This script checks dependencies and launches the Streamlit app
"""

import subprocess
import sys
import os

def check_and_install_dependencies():
    """Check if required packages are installed, install if missing"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'yfinance',
        'scikit-learn',
        'plotly',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing packages: {e}")
            return False
    
    return True

def main():
    """Main function to setup and run the application"""
    print("🚀 Streamlit Trading Strategy Application Setup")
    print("=" * 50)
    
    # Check dependencies
    if not check_and_install_dependencies():
        print("\n❌ Failed to install dependencies. Please install manually:")
        print("pip install -r requirements.txt")
        return
    
    # Check if streamlit app exists
    if not os.path.exists("streamlit_trading_app.py"):
        print("\n❌ Error: streamlit_trading_app.py not found!")
        print("Make sure you're in the correct directory.")
        return
    
    print("\n🌟 Starting Streamlit application...")
    print("=" * 50)
    print("📱 The application will open in your web browser")
    print("🔗 If not automatically opened, navigate to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_trading_app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")

if __name__ == "__main__":
    main()