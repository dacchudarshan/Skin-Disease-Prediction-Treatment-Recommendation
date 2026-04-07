#!/usr/bin/env python3
"""
Performance Optimization Toolkit for Skin Disease Classifier
Measures startup time and implements quick optimization strategies
"""

import time
import os
import sys
import py_compile
import subprocess
from pathlib import Path

class PerformanceOptimizer:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self.results = {}
        
    def measure_startup_time(self):
        """Measure current app startup time"""
        print("\n📊 Measuring current startup time...")
        
        cmd = "cd {} && time python -c \"from app import app; print('App loaded')\"".format(self.project_dir)
        start = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        elapsed = time.time() - start
        
        self.results['baseline_startup'] = elapsed
        print(f"✅ Baseline startup time: {elapsed:.2f}s")
        return elapsed
    
    def compile_bytecode(self):
        """Compile Python files to bytecode (.pyc)"""
        print("\n🔧 Compiling Python files to bytecode...")
        
        py_files = list(self.project_dir.glob('*.py'))
        compiled = 0
        
        for py_file in py_files:
            try:
                py_compile.compile(str(py_file), doraise=True)
                print(f"  ✅ {py_file.name}")
                compiled += 1
            except Exception as e:
                print(f"  ❌ {py_file.name}: {e}")
        
        self.results['compiled_files'] = compiled
        print(f"\n✅ Compiled {compiled} Python files to bytecode")
        return compiled
    
    def create_optimized_app(self):
        """Create version with lazy imports"""
        print("\n🚀 Creating optimized app version...")
        
        optimization_code = '''
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
'''
        
        opt_file = self.project_dir / 'app_optimized.py'
        with open(opt_file, 'w') as f:
            f.write(optimization_code)
        
        print(f"✅ Created optimized app: {opt_file.name}")
        return str(opt_file)
    
    def generate_performance_report(self):
        """Generate performance measurement report"""
        print("\n📈 Performance Report")
        print("=" * 60)
        
        for metric, value in self.results.items():
            if isinstance(value, float):
                print(f"  {metric:.<40} {value:>10.2f}s")
            else:
                print(f"  {metric:.<40} {value:>10}")
        
        print("=" * 60)
        
        # Calculate improvements
        if 'baseline_startup' in self.results:
            baseline = self.results['baseline_startup']
            target = baseline * 0.5  # 50% improvement target
            print(f"\n🎯 Target startup time: {target:.2f}s (50% faster)")
            print(f"💾 Files compiled: {self.results.get('compiled_files', 0)}")
        
        return self.results
    
    def create_gunicorn_config(self):
        """Create Gunicorn configuration for production"""
        print("\n⚙️  Creating Gunicorn configuration...")
        
        gunicorn_config = '''
# Gunicorn configuration for Skin Disease Classifier
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"

# Process naming
proc_name = "skin-disease-classifier"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None

# SSL
keyfile = None
certfile = None

# Application
preload_app = False
paste = None
'''
        
        config_file = self.project_dir / 'gunicorn_config.py'
        with open(config_file, 'w') as f:
            f.write(gunicorn_config)
        
        print(f"✅ Created Gunicorn config: {config_file.name}")
        
        # Create logs directory
        os.makedirs(self.project_dir / 'logs', exist_ok=True)
        print("✅ Created logs directory")
        
        return str(config_file)
    
    def create_run_scripts(self):
        """Create convenient run scripts"""
        print("\n📝 Creating run scripts...")
        
        # Development script
        dev_script = '''#!/bin/bash
echo "🚀 Starting Skin Disease Classifier (Development)"
python app.py
'''
        
        # Production script
        prod_script = '''#!/bin/bash
echo "🚀 Starting Skin Disease Classifier (Production with Gunicorn)"
pip install gunicorn -q 2>/dev/null
gunicorn -c gunicorn_config.py app:app
'''
        
        # Optimized script
        opt_script = '''#!/bin/bash
echo "⚡ Starting Skin Disease Classifier (Optimized)"
python -m compileall .
echo "Running with optimized settings..."
gunicorn -w 4 -b 0.0.0.0:8000 --timeout 30 app:app
'''
        
        dev_file = self.project_dir / 'run_dev.sh'
        prod_file = self.project_dir / 'run_prod.sh'
        opt_file = self.project_dir / 'run_optimized.sh'
        
        with open(dev_file, 'w') as f:
            f.write(dev_script)
        os.chmod(dev_file, 0o755)
        
        with open(prod_file, 'w') as f:
            f.write(prod_script)
        os.chmod(prod_file, 0o755)
        
        with open(opt_file, 'w') as f:
            f.write(opt_script)
        os.chmod(opt_file, 0o755)
        
        print(f"✅ Created run_dev.sh")
        print(f"✅ Created run_prod.sh")
        print(f"✅ Created run_optimized.sh")
        
        return {
            'dev': str(dev_file),
            'prod': str(prod_file),
            'optimized': str(opt_file)
        }
    
    def run_optimization(self):
        """Run full optimization suite"""
        print("\n" + "="*60)
        print("🚀 SKIN DISEASE CLASSIFIER - PERFORMANCE OPTIMIZATION")
        print("="*60)
        
        # Step 1: Measure baseline
        self.measure_startup_time()
        
        # Step 2: Compile bytecode
        self.compile_bytecode()
        
        # Step 3: Create optimized version
        self.create_optimized_app()
        
        # Step 4: Create Gunicorn config
        self.create_gunicorn_config()
        
        # Step 5: Create run scripts
        self.create_run_scripts()
        
        # Step 6: Generate report
        self.generate_performance_report()
        
        print("\n✅ Optimization complete!")
        print("\n📋 Next steps:")
        print("  1. ./run_optimized.sh     - Run with optimizations")
        print("  2. ./run_prod.sh          - Run with Gunicorn (production)")
        print("  3. Check PERFORMANCE_GUIDE.md for more details")


if __name__ == '__main__':
    project_dir = Path(__file__).parent
    optimizer = PerformanceOptimizer(project_dir)
    optimizer.run_optimization()
