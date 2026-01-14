#!/usr/bin/env python
"""
Customize TransitKit with your information.
Run: python customize.py
"""

import os
import re

def update_file(filepath, changes):
    """Update a file with changes."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in changes.items():
        content = content.replace(old, new)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✓ Updated {filepath}")

def main():
    print("=== TransitKit Customization ===\n")
    
    # Get your info
    your_name = input("Your full name: ").strip()
    your_email = input("Your email: ").strip()
    github_user = input("Your GitHub username: ").strip()
    
    # Define what to change
    changes = {
        "YOUR NAME": your_name,
        "YOUR@EMAIL.COM": your_email,
        "YOURUSERNAME": github_user,
    }
    
    # Files to update
    files = [
        "pyproject.toml",
        "src/transitkit/__init__.py",
        "README.md",
    ]
    
    # Update each file
    for file in files:
        if os.path.exists(file):
            update_file(file, changes)
    
    print("\n✅ Customization complete!")
    print(f"\nYour GitHub repository will be at:")
    print(f"https://github.com/{github_user}/transitkit")

if __name__ == "__main__":
    main()