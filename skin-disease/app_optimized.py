
# ============================================
# OPTIMIZED IMPORTS FOR FASTER STARTUP
# ============================================

import os
import sys
from pathlib import Path
from datetime import datetime

# Fast-load core dependencies
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Lazy imports for heavy dependencies
REPORTLAB_AVAILABLE = False
TENSORFLOW_AVAILABLE = False
PIL_AVAILABLE = False

# Lazy load on first use
_reportlab = None
_tensorflow = None
_pil_image = None
_pil_stat = None

def get_reportlab():
    """Lazy load reportlab on first use"""
    global _reportlab, REPORTLAB_AVAILABLE
    if _reportlab is None:
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            REPORTLAB_AVAILABLE = True
            return {
                'letter': letter, 'A4': A4, 'colors': colors,
                'inch': inch, 'SimpleDocTemplate': SimpleDocTemplate
            }
        except ImportError:
            REPORTLAB_AVAILABLE = False
            return None
    return _reportlab

def get_pil():
    """Lazy load PIL on first use"""
    global _pil_image, _pil_stat, PIL_AVAILABLE
    if _pil_image is None:
        try:
            from PIL import Image, ImageStat
            PIL_AVAILABLE = True
            _pil_image = Image
            _pil_stat = ImageStat
        except ImportError:
            PIL_AVAILABLE = False
    return _pil_image, _pil_stat

# ============================================
# FAST INITIALIZATION
# ============================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Fast cache for disease info
_DISEASE_INFO_CACHE = None

def get_disease_info_cached(disease_name):
    """Get disease info from fast cache"""
    global _DISEASE_INFO_CACHE
    if _DISEASE_INFO_CACHE is None:
        _DISEASE_INFO_CACHE = load_disease_info_fast()
    return _DISEASE_INFO_CACHE.get(disease_name, {})

def load_disease_info_fast():
    """Load disease info efficiently"""
    # Replace with actual disease info loading
    return {
        'acne': {'description': 'Common skin condition'},
        'eczema': {'description': 'Inflammatory skin condition'},
    }

print("✅ App initialized in optimized mode")
