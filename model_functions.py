#!/usr/bin/env python3
"""
Model functions module
Contains the negbinner and stanbinner functions for import by other scripts
"""

# Import the functions from the main models file
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the actual 6_final_models.py file
exec(open('6_final_models.py').read().split('if __name__')[0])
# Functions negbinner and stanbinner are now available