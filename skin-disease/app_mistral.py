"""
Lightweight Mistral Vision API Server
Runs without TensorFlow/Keras for quick testing
"""

import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import Mistral Vision
from mistral_vision import MistralVisionAnalyzer, BatchMistralAnalyzer

# Create Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mistral Vision API - Skin Disease Detection</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 50px; background: #f5f5f5; }
            .container { background: white; padding: 40px; border-radius: 10px; max-width: 600px; margin: 0 auto; }
            h1 { color: #667eea; }
            .button { background: #667eea; color: white; padding: 12px 30px; border: none; 
                     border-radius: 6px; font-size: 16px; cursor: pointer; margin: 10px; }
            .button:hover { background: #764ba2; }
            .info { background: #e8f5e9; padding: 15px; border-radius: 6px; margin: 20px 0; }
            code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Skin Disease Detection</h1>
            <p>Using Mistral Vision API for AI-Powered Analysis</p>
            
            <div class="info">
                <p><strong>✅ Server is Running!</strong></p>
            </div>
            
            <p>
                <a href="/mistral"><button class="button">📸 Start Analysis</button></a>
                <a href="/api-docs"><button class="button">📚 API Docs</button></a>
            </p>
            
            <hr style="margin: 30px 0;">
            
            <h3>Available Endpoints:</h3>
            <ul style="text-align: left; display: inline-block;">
                <li><strong>GET /mistral</strong> - Web interface</li>
                <li><strong>POST /mistral-analyze</strong> - Single image analysis</li>
                <li><strong>POST /mistral-batch</strong> - Batch analysis</li>
                <li><strong>GET /api-docs</strong> - API documentation</li>
                <li><strong>GET /health</strong> - Health check</li>
            </ul>
        </div>
    </body>
    </html>
    '''

@app.route('/mistral')
def mistral_page():
    """Mistral Vision API page."""
    return render_template('mistral_analysis.html')

@app.route('/api-docs')
def api_docs():
    """API documentation."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mistral Vision API - Documentation</title>
        <style>
            body { font-family: 'Courier New'; background: #1e1e1e; color: #d4d4d4; padding: 20px; }
            .container { max-width: 1000px; margin: 0 auto; }
            h1, h2 { color: #667eea; }
            .endpoint { background: #2d2d2d; padding: 15px; margin: 20px 0; border-left: 4px solid #667eea; }
            code { background: #1e1e1e; padding: 10px; display: block; margin: 10px 0; }
            pre { overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔌 Mistral Vision API - Documentation</h1>
            
            <div class="endpoint">
                <h2>POST /mistral-analyze</h2>
                <p>Analyze a single skin image</p>
                <code>
curl -X POST http://localhost:5000/mistral-analyze \\
  -F "file=@image.jpg"
                </code>
            </div>
            
            <div class="endpoint">
                <h2>POST /mistral-batch</h2>
                <p>Analyze multiple images</p>
                <code>
curl -X POST http://localhost:5000/mistral-batch \\
  -F "files=@image1.jpg" \\
  -F "files=@image2.jpg"
                </code>
            </div>
            
            <div class="endpoint">
                <h2>GET /health</h2>
                <p>Server health check</p>
                <code>curl http://localhost:5000/health</code>
            </div>
            
            <h2>Response Example</h2>
            <code>
{
  "success": true,
  "accuracy_confidence": 87.5,
  "analysis": {
    "disease": "Melanoma",
    "confidence": 87.5,
    "severity": "High"
  }
}
            </code>
        </div>
    </body>
    </html>
    '''

@app.route('/mistral-analyze', methods=['POST'])
def mistral_analyze():
    """Single image analysis with Mistral."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze with Mistral
        analyzer = MistralVisionAnalyzer()
        result = analyzer.analyze_skin_condition(filepath)
        
        # Add preview
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        result['preview'] = f'data:image/png;base64,{img_data}'
        result['filename'] = filename
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/mistral-batch', methods=['POST'])
def mistral_batch():
    """Batch image analysis."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        filepaths = []
        for file in files:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                filepaths.append(filepath)
        
        if not filepaths:
            return jsonify({'error': 'No valid files'}), 400
        
        # Batch analyze
        batch = BatchMistralAnalyzer()
        results = batch.analyze_batch(filepaths, show_progress=False)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health')
def health():
    """Health check."""
    return jsonify({
        'status': 'healthy',
        'service': 'Mistral Vision API',
        'mistral_available': True
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Mistral Vision API - Skin Disease Detection")
    print("=" * 60)
    print("\n✅ Server Starting...")
    print("🌐 Web Interface: http://localhost:5000/mistral")
    print("📚 API Docs: http://localhost:5000/api-docs")
    print("❤️  Health Check: http://localhost:5000/health")
    print("\n" + "=" * 60)
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=True, use_reloader=False, port=5000)
