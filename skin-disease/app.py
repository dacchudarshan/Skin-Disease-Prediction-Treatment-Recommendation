import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image, ImageStat
import base64
import zipfile
import io
from pathlib import Path
from datetime import datetime
import json
import cv2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("reportlab not installed. PDF export feature will not be available.")

# Try to import tensorflow/keras for ML model
# Disabled due to macOS compatibility issues
try:
    import os as _os_check
    if _os_check.uname().sysname != 'Darwin':  # Only load on non-macOS
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image
        TENSORFLOW_AVAILABLE = True
    else:
        TENSORFLOW_AVAILABLE = False
        print("TensorFlow disabled on macOS to avoid compatibility issues.")
except Exception:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using image analysis for classification.")

# Import advanced accuracy module
try:
    from advanced_accuracy import AdvancedDiseaseDetector, AccuracyEnhancer, NumpyEncoder, convert_numpy_types
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False
    print("Advanced accuracy module not available. Using basic analysis.")

# Import Mistral Vision API for enhanced accuracy
try:
    from mistral_vision import MistralVisionAnalyzer, BatchMistralAnalyzer
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("Mistral Vision API not available. Enhanced accuracy analysis will be disabled.")

# ===========================
# CONFIGURATION
# ===========================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure JSON encoder for numpy types
if ADVANCED_ANALYSIS_AVAILABLE:
    app.json_encoder = NumpyEncoder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# ===========================
# HELPER FUNCTIONS
# ===========================
def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_disease_information(disease_name):
    """Get detailed information and recommendations for detected disease."""
    disease_db = {
        'Eczema/Dermatitis': {
            'severity': 'Medium',
            'description': 'Inflammatory skin condition causing itching, redness, and irritation.',
            'symptoms': ['Red, inflamed skin', 'Intense itching', 'Dry, sensitive skin', 'Small raised bumps'],
            'recommendations': [
                'Keep skin moisturized with fragrance-free lotions',
                'Avoid harsh soaps and hot water',
                'Identify and avoid triggers',
                'Consult a dermatologist for prescription treatments',
                'Consider anti-inflammatory creams'
            ],
            'urgency': 'Non-urgent'
        },
        'Psoriasis': {
            'severity': 'Medium-High',
            'description': 'Chronic autoimmune condition causing thick, scaly patches.',
            'symptoms': ['Red, scaly patches', 'Silvery scales', 'Burning sensation', 'Joint pain'],
            'recommendations': [
                'Apply moisturizer regularly',
                'Use prescribed topical treatments',
                'Manage stress levels',
                'Avoid skin injury',
                'See a dermatologist for systemic therapy options'
            ],
            'urgency': 'Non-urgent but requires management'
        },
        'Melanoma/Mole (High Risk)': {
            'severity': 'High',
            'description': 'Potentially serious skin cancer. Any dark spots should be professionally evaluated.',
            'symptoms': ['Dark brown or black spots', 'Asymmetrical shape', 'Irregular borders', 'Size > 6mm'],
            'recommendations': [
                '⚠️ URGENT: Schedule appointment with dermatologist immediately',
                'Do NOT wait - early detection is critical',
                'Get professional biopsy if recommended',
                'Monitor all moles regularly',
                'Practice sun protection (SPF 30+)'
            ],
            'urgency': 'URGENT - See doctor immediately'
        },
        'Acne': {
            'severity': 'Low-Medium',
            'description': 'Common skin condition caused by clogged pores and bacteria.',
            'symptoms': ['Pimples', 'Blackheads', 'Whiteheads', 'Oily skin'],
            'recommendations': [
                'Wash face twice daily with gentle cleanser',
                'Use non-comedogenic moisturizer',
                'Apply benzoyl peroxide or salicylic acid',
                'Avoid touching or squeezing lesions',
                'Consider dermatologist visit for severe acne'
            ],
            'urgency': 'Non-urgent'
        },
        'Benign Keratosis/Nevus': {
            'severity': 'Low',
            'description': 'Common, non-cancerous skin growths. Usually harmless.',
            'symptoms': ['Brown or tan spots', 'Waxy or scaly appearance', 'Well-defined borders'],
            'recommendations': [
                'Monitor for changes in size, color, or shape',
                'Can be removed for cosmetic reasons',
                'No treatment necessary if benign',
                'Annual skin check recommended',
                'Removal can be done by dermatologist'
            ],
            'urgency': 'Non-urgent'
        },
        'Normal/Healthy Skin': {
            'severity': 'None',
            'description': 'Your skin appears to be in good condition.',
            'symptoms': [],
            'recommendations': [
                'Maintain good skincare routine',
                'Use sunscreen daily (SPF 30+)',
                'Stay hydrated',
                'Get regular sleep',
                'Consider annual dermatology check'
            ],
            'urgency': 'No immediate action needed'
        }
    }
    
    # Return disease info or default
    info = disease_db.get(disease_name, disease_db['Normal/Healthy Skin'])
    return info

def extract_image_features(image_path):
    """Extract features from image for disease classification."""
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate image statistics
        stat = ImageStat.Stat(img)
        avg_color = stat.mean[:3]  # R, G, B averages
        
        # Calculate color ratios
        r_ratio = avg_color[0] / 255.0
        g_ratio = avg_color[1] / 255.0
        b_ratio = avg_color[2] / 255.0
        
        return {
            'r': r_ratio,
            'g': g_ratio,
            'b': b_ratio,
            'redness': r_ratio - g_ratio,  # High = reddish
            'darkness': 1 - (sum(avg_color) / (3 * 255))  # High = dark
        }
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def classify_skin_disease(features):
    """Classify skin disease based on image features."""
    if not features:
        return {'disease': 'Unknown', 'confidence': 0.0}
    
    redness = features['redness']
    darkness = features['darkness']
    r = features['r']
    g = features['g']
    
    # Disease classification heuristics based on color analysis
    if redness > 0.15 and r > 0.6:
        # Reddish inflammation - likely Eczema or Dermatitis
        confidence = min(95, 70 + (redness * 50))
        return {'disease': 'Eczema/Dermatitis', 'confidence': round(confidence, 1)}
    
    elif redness > 0.1 and darkness > 0.3:
        # Dark and reddish - likely Psoriasis
        confidence = min(95, 75 + (darkness * 30))
        return {'disease': 'Psoriasis', 'confidence': round(confidence, 1)}
    
    elif darkness > 0.5:
        # Very dark spots - likely Melanoma or Mole
        confidence = min(90, 60 + (darkness * 40))
        return {'disease': 'Melanoma/Mole (High Risk)', 'confidence': round(confidence, 1)}
    
    elif redness > 0.05 and 0.1 < darkness < 0.4:
        # Mild inflammation - likely Acne
        confidence = min(85, 65 + (redness * 40))
        return {'disease': 'Acne', 'confidence': round(confidence, 1)}
    
    elif darkness > 0.15 and darkness < 0.3:
        # Medium darkness - likely Benign Keratosis or Nevus
        confidence = min(80, 70 + (darkness * 20))
        return {'disease': 'Benign Keratosis/Nevus', 'confidence': round(confidence, 1)}
    
    else:
        # Relatively normal appearance
        return {'disease': 'Normal/Healthy Skin', 'confidence': 70.0}

def load_model():
    """Load the pre-trained skin disease model."""
    model_path = 'models/skin_model.h5'
    if os.path.exists(model_path) and TENSORFLOW_AVAILABLE:
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

def predict_disease(image_path, model=None):
    """Predict skin disease from image using advanced multi-method analysis."""
    try:
        # First priority: Use advanced accuracy enhancement
        if ADVANCED_ANALYSIS_AVAILABLE:
            try:
                enhancer = AccuracyEnhancer()
                advanced_result = enhancer.enhance_prediction(image_path)
                
                if advanced_result.get("disease_predictions"):
                    top_prediction = advanced_result["disease_predictions"][0]
                    return {
                        'disease': top_prediction['disease'],
                        'confidence': round(top_prediction['confidence'] * 100, 2),
                        'method': 'Advanced Multi-Model Analysis',
                        'all_predictions': advanced_result['disease_predictions'],
                        'severity': advanced_result.get('severity_assessment', {}).get('level'),
                        'recommendations': advanced_result.get('recommendations', []),
                        'accuracy_level': advanced_result.get('accuracy_level'),
                        'analysis_details': {
                            'color_analysis': advanced_result['raw_analysis'].get('color_analysis', {}),
                            'texture_analysis': advanced_result['raw_analysis'].get('texture_analysis', {}),
                            'morphology_analysis': advanced_result['raw_analysis'].get('morphology_analysis', {}),
                            'edge_analysis': advanced_result['raw_analysis'].get('edge_analysis', {})
                        }
                    }
            except Exception as e:
                print(f"Advanced analysis error: {e}. Falling back to alternative methods.")
        
        # Second priority: TensorFlow model
        if TENSORFLOW_AVAILABLE and model is not None:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            predictions = model.predict(img_array, verbose=0)
            confidence = float(np.max(predictions[0]))
            predicted_class = int(np.argmax(predictions[0]))
            
            class_names = [
                'Melanoma',
                'Nevus',
                'Basal Cell Carcinoma',
                'Actinic Keratosis',
                'Benign Keratosis',
                'Dermatofibroma',
                'Vascular Lesion'
            ]
            
            disease = class_names[predicted_class] if predicted_class < len(class_names) else 'Unknown'
            
            return {
                'disease': disease,
                'confidence': round(confidence * 100, 2),
                'class_index': predicted_class,
                'method': 'Deep Learning Model',
                'accuracy_level': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'
            }
        
        # Third priority: Feature-based classification
        else:
            features = extract_image_features(image_path)
            if features:
                result = classify_skin_disease(features)
                result['method'] = 'Image Analysis'
                result['accuracy_level'] = 'Medium'
                return result
            else:
                return {
                    'disease': 'Unable to analyze',
                    'confidence': 0.0,
                    'message': 'Could not process image'
                }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'disease': 'Error',
            'confidence': 0.0,
            'message': str(e)
        }

def predict_with_mistral(image_path):
    """Predict skin disease using Mistral Vision API with accuracy metrics."""
    try:
        if not MISTRAL_AVAILABLE:
            return None
        
        analyzer = MistralVisionAnalyzer()
        result = analyzer.analyze_skin_condition(image_path)
        
        if result.get('success'):
            analysis = result.get('analysis', {})
            return {
                'disease': analysis.get('disease', 'Unknown'),
                'confidence': analysis.get('confidence', 0),
                'method': 'Mistral Vision API',
                'severity': analysis.get('severity', 'Unknown'),
                'observations': analysis.get('observations', []),
                'recommendations': analysis.get('recommendations', []),
                'accuracy_metrics': analysis.get('accuracy_metrics', {}),
                'accuracy_level': 'High' if analysis.get('confidence', 0) > 80 else 'Medium' if analysis.get('confidence', 0) > 60 else 'Low',
                'accuracy': analysis.get('confidence', 0)
            }
        else:
            return None
    except Exception as e:
        print(f"Mistral analysis error: {e}")
        return None

# ===========================
# TIER 1 ADVANCED FEATURES
# ===========================

def get_confidence_category(confidence):
    """Categorize confidence level with reasoning."""
    if confidence < 30:
        return {
            'category': 'Low Confidence',
            'color': '#FF5252',
            'reasoning': 'Image clarity or features are ambiguous. Consider retaking with better lighting.',
            'recommendation': 'Retake image with natural lighting and clear focus'
        }
    elif confidence < 60:
        return {
            'category': 'Medium Confidence',
            'color': '#FFC107',
            'reasoning': 'Detection is reasonably reliable but uncertainty exists.',
            'recommendation': 'Consider professional medical evaluation for confirmation'
        }
    elif confidence < 85:
        return {
            'category': 'High Confidence',
            'color': '#66BB6A',
            'reasoning': 'Strong detection with good image quality and features.',
            'recommendation': 'Results are reliable for initial assessment'
        }
    else:
        return {
            'category': 'Very High Confidence',
            'color': '#2E7D32',
            'reasoning': 'Excellent image quality and very strong feature match.',
            'recommendation': 'Results are highly reliable'
        }

def calculate_abcde_score(image_path):
    """Calculate ABCDE Melanoma Risk Score."""
    try:
        img = Image.open(image_path)
        img_resized = img.resize((224, 224))
        
        if img_resized.mode != 'RGB':
            img_resized = img_resized.convert('RGB')
        
        # Get image statistics
        stat = ImageStat.Stat(img_resized)
        avg_color = stat.mean[:3]
        pixels = list(img_resized.getdata())
        
        # Simple heuristic scoring (0-5 scale)
        scores = {}
        
        # A - Asymmetry (check if edges are symmetric)
        # Simplified: check color distribution
        r_ratio = avg_color[0] / 255.0
        scores['asymmetry'] = min(2, int(abs(r_ratio - 0.5) * 4))
        
        # B - Border (irregular borders)
        # Simplified: check color variance
        color_std = np.std([p[:3] for p in pixels[:100]])
        scores['border'] = min(2, int(color_std / 50))
        
        # C - Color (multiple colors)
        # Count distinct colors
        unique_colors = len(set([tuple(p[:3]) for p in pixels[::10]]))
        scores['color'] = min(2, int(unique_colors / 100))
        
        # D - Diameter (size > 6mm is concerning)
        img_size = img.size[0]
        scores['diameter'] = 2 if img_size > 224 else 1
        
        # E - Elevation (raised or flat)
        # Simplified: check darkness
        darkness = 1 - (sum(avg_color) / (3 * 255))
        scores['elevation'] = min(2, int(darkness * 3))
        
        total_score = sum(scores.values())
        risk_level = {
            0: {'level': 'Very Low', 'color': '#2E7D32', 'action': 'Continue monitoring'},
            1: {'level': 'Low', 'color': '#66BB6A', 'action': 'Annual check-up'},
            2: {'level': 'Moderate', 'color': '#FBC02D', 'action': 'Professional evaluation recommended'},
            3: {'level': 'High', 'color': '#F57C00', 'action': 'Seek dermatologist soon'},
            4: {'level': 'Very High', 'color': '#D32F2F', 'action': 'URGENT: See dermatologist immediately'},
        }
        
        return {
            'scores': scores,
            'total': total_score,
            'risk': risk_level.get(min(total_score, 4), risk_level[4])
        }
    except Exception as e:
        print(f"ABCDE Score calculation error: {e}")
        return None

def enhance_image(image_path, brightness=1.0, contrast=1.0, color_balance=1.0):
    """Enhance image for better analysis."""
    try:
        from PIL import ImageEnhance
        
        img = Image.open(image_path)
        
        # Apply brightness enhancement
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
        
        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
        
        # Apply color enhancement
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(color_balance)
        
        # Save enhanced image
        enhanced_path = image_path.replace('.', '_enhanced.')
        img.save(enhanced_path)
        
        return {
            'success': True,
            'original_path': image_path,
            'enhanced_path': enhanced_path,
            'settings': {
                'brightness': brightness,
                'contrast': contrast,
                'color_balance': color_balance
            }
        }
    except Exception as e:
        print(f"Image enhancement error: {e}")
        return {'success': False, 'error': str(e)}

def get_similar_cases(disease_name, confidence):
    """Get similar historical cases from database."""
    # Sample database of similar cases
    similar_cases_db = {
        'Eczema/Dermatitis': [
            {'id': 1, 'disease': 'Eczema/Dermatitis', 'similarity': 92, 'severity': 'Medium', 'status': 'Improved with treatment'},
            {'id': 2, 'disease': 'Eczema/Dermatitis', 'similarity': 88, 'severity': 'Medium', 'status': 'Ongoing treatment'},
            {'id': 3, 'disease': 'Eczema/Dermatitis', 'similarity': 85, 'severity': 'High', 'status': 'Controlled with therapy'},
        ],
        'Psoriasis': [
            {'id': 4, 'disease': 'Psoriasis', 'similarity': 90, 'severity': 'Medium-High', 'status': 'Managed'},
            {'id': 5, 'disease': 'Psoriasis', 'similarity': 87, 'severity': 'High', 'status': 'Under treatment'},
            {'id': 6, 'disease': 'Psoriasis', 'similarity': 84, 'severity': 'Medium', 'status': 'Improving'},
        ],
        'Melanoma/Mole (High Risk)': [
            {'id': 7, 'disease': 'Melanoma/Mole', 'similarity': 95, 'severity': 'High', 'status': 'Early detection - positive outcome'},
            {'id': 8, 'disease': 'Mole', 'similarity': 89, 'severity': 'Low', 'status': 'Benign - monitored'},
            {'id': 9, 'disease': 'Melanoma/Mole', 'similarity': 86, 'severity': 'High', 'status': 'Post-treatment follow-up'},
        ],
        'Acne': [
            {'id': 10, 'disease': 'Acne', 'similarity': 91, 'severity': 'Medium', 'status': 'Cleared with medication'},
            {'id': 11, 'disease': 'Acne', 'similarity': 88, 'severity': 'Low-Medium', 'status': 'Ongoing management'},
            {'id': 12, 'disease': 'Acne', 'similarity': 85, 'severity': 'Medium', 'status': 'Improving with topical treatment'},
        ]
    }
    
    return {
        'cases': similar_cases_db.get(disease_name, []),
        'disease': disease_name,
        'confidence': confidence
    }

# ===========================
# ROUTES
# ===========================
@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/mistral')
def mistral_page():
    """Mistral Vision API analysis page."""
    if MISTRAL_AVAILABLE:
        return render_template('mistral_analysis.html')
    else:
        return jsonify({'error': 'Mistral Vision API not configured'}), 503

@app.route('/advanced-features')
def advanced_features():
    """Advanced features roadmap page."""
    return render_template('advanced_features.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'model_exists': os.path.exists('models/skin_model.h5')
    })

@app.route('/gallery')
def gallery():
    """View all test images."""
    return render_template('gallery.html')

@app.route('/batch-analysis')
def batch_analysis():
    """Batch image analysis page."""
    return render_template('batch_analysis.html')

@app.route('/statistics')
def statistics():
    """Statistics and analytics dashboard."""
    return render_template('statistics.html')

@app.route('/comparison')
def comparison():
    """Image comparison page for side-by-side analysis."""
    return render_template('comparison.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction with Mistral Vision API."""
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if file has filename
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Try Mistral Vision API first (if available)
        result = None
        if MISTRAL_AVAILABLE:
            result = predict_with_mistral(filepath)
            if result:
                result['method'] = 'Mistral Vision API (AI-Powered)'
        
        # Fallback to standard analysis if Mistral not available
        if not result:
            model = load_model()
            result = predict_disease(filepath, model)
        
        # Convert numpy types to native Python types
        if ADVANCED_ANALYSIS_AVAILABLE:
            result = convert_numpy_types(result)
        
        # Add disease information and recommendations if not from Mistral
        if result.get('method') != 'Mistral Vision API (AI-Powered)':
            disease_info = get_disease_information(result['disease'])
            result.update(disease_info)
        
        # Read image for preview
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        result['preview'] = f'data:image/png;base64,{img_data}'
        result['filename'] = filename
        
        # Final conversion to ensure JSON serializable
        result = convert_numpy_types(result) if ADVANCED_ANALYSIS_AVAILABLE else result
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/mistral-analyze', methods=['POST'])
def mistral_analyze():
    """Analyze skin disease using Mistral Vision API for enhanced accuracy."""
    if not MISTRAL_AVAILABLE:
        return jsonify({'error': 'Mistral Vision API not configured'}), 503
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if file has filename
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize Mistral Vision Analyzer
        analyzer = MistralVisionAnalyzer()
        
        # Analyze with Mistral
        mistral_result = analyzer.analyze_skin_condition(filepath)
        
        # Read image for preview
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        mistral_result['preview'] = f'data:image/png;base64,{img_data}'
        mistral_result['filename'] = filename
        
        # If analysis succeeded, get additional disease information
        if mistral_result.get('success'):
            disease_name = mistral_result.get('analysis', {}).get('disease', 'Unknown')
            disease_info = get_disease_information(disease_name)
            if disease_info:
                mistral_result['disease_info'] = disease_info
        
        return jsonify(mistral_result)
    
    except Exception as e:
        return jsonify({
            'error': f'Mistral analysis error: {str(e)}',
            'method': 'Mistral Vision API',
            'success': False
        }), 500

@app.route('/mistral-batch', methods=['POST'])
def mistral_batch():
    """Batch analyze multiple images with Mistral Vision API."""
    if not MISTRAL_AVAILABLE:
        return jsonify({'error': 'Mistral Vision API not configured'}), 503
    
    # Check if files are in request
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        # Save uploaded files
        filepaths = []
        for file in files:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                filepaths.append(filepath)
        
        if not filepaths:
            return jsonify({'error': 'No valid files found'}), 400
        
        # Initialize batch analyzer
        batch_analyzer = BatchMistralAnalyzer()
        
        # Process batch
        batch_result = batch_analyzer.analyze_batch(filepaths, show_progress=False)
        
        return jsonify(batch_result)
    
    except Exception as e:
        return jsonify({
            'error': f'Batch analysis error: {str(e)}',
            'success': False
        }), 500

@app.route('/disease-info')
def disease_info():
    """Disease information page."""
    diseases = {
        'Normal/Healthy Skin': {
            'emoji': '✅',
            'description': 'Your skin appears to be in good condition.',
            'symptoms': ['Clear skin', 'No visible lesions', 'Normal color and texture'],
            'causes': ['Genetic factors', 'Good skincare routine', 'Healthy lifestyle'],
            'treatments': ['Maintain good hygiene', 'Use sunscreen', 'Balanced diet'],
            'prevention': ['Daily sunscreen (SPF 30+)', 'Moisturize regularly', 'Get adequate sleep', 'Drink water'],
            'when_to_see_doctor': 'Annual dermatology check recommended'
        },
        'Acne': {
            'emoji': '🟡',
            'description': 'Acne is a common skin condition characterized by pimples, blackheads, and whiteheads.',
            'symptoms': ['Pimples', 'Blackheads', 'Whiteheads', 'Oily skin', 'Mild inflammation'],
            'causes': ['Bacterial growth', 'Excess sebum', 'Dead skin cells', 'Hormonal changes'],
            'treatments': ['Topical treatments (benzoyl peroxide, salicylic acid)', 'Antibiotics', 'Retinoids', 'Professional extraction'],
            'prevention': ['Clean face twice daily', 'Don\'t squeeze pimples', 'Use oil-free products', 'Manage stress'],
            'when_to_see_doctor': 'Consult dermatologist if severe or not responding to treatment'
        },
        'Eczema/Dermatitis': {
            'emoji': '🔴',
            'description': 'Eczema is an inflammatory condition causing itching, redness, and irritation.',
            'symptoms': ['Intense itching', 'Red patches', 'Dry skin', 'Small raised bumps', 'Cracked skin'],
            'causes': ['Genetic predisposition', 'Allergens', 'Irritants', 'Stress', 'Weather changes'],
            'treatments': ['Moisturizers', 'Topical corticosteroids', 'Antihistamines', 'Avoid irritants'],
            'prevention': ['Use mild soaps', 'Moisturize daily', 'Avoid irritants', 'Manage stress', 'Control humidity'],
            'when_to_see_doctor': 'See dermatologist if symptoms persist or worsen'
        },
        'Psoriasis': {
            'emoji': '🟠',
            'description': 'Psoriasis is an autoimmune condition causing scaly, inflamed patches.',
            'symptoms': ['Red patches', 'Silvery scales', 'Itching', 'Burning sensation', 'Thick nails'],
            'causes': ['Genetic factors', 'Immune system dysfunction', 'Stress', 'Infections', 'Medications'],
            'treatments': ['Topical treatments', 'Phototherapy', 'Systemic medications', 'Biologics'],
            'prevention': ['Manage stress', 'Avoid triggers', 'Maintain good hygiene', 'Use moisturizers'],
            'when_to_see_doctor': 'Dermatologist visit is important for proper diagnosis and treatment'
        },
        'Melanoma/Mole (High Risk)': {
            'emoji': '⚫',
            'description': 'Melanoma is the most serious type of skin cancer. Moles require monitoring.',
            'symptoms': ['Dark spots', 'Asymmetry', 'Irregular borders', 'Color variation', 'Size changes'],
            'causes': ['UV exposure', 'Genetic predisposition', 'Fair skin', 'Previous sunburns', 'Mole history'],
            'treatments': ['Surgical removal', 'Chemotherapy', 'Immunotherapy', 'Radiation therapy'],
            'prevention': ['Use sunscreen (SPF 50+)', 'Avoid sun exposure', 'Wear protective clothing', 'Regular skin checks', 'Avoid tanning beds'],
            'when_to_see_doctor': '⚠️ URGENT - See dermatologist immediately if suspected'
        },
        'Benign Keratosis/Nevus': {
            'emoji': '🟤',
            'description': 'Benign skin growths that are non-cancerous but should be monitored.',
            'symptoms': ['Brown spots', 'Raised bumps', 'Waxy appearance', 'Well-defined borders'],
            'causes': ['Age', 'Sun exposure', 'Genetic factors'],
            'treatments': ['No treatment needed (if benign)', 'Removal for cosmetic reasons', 'Cryotherapy', 'Laser removal'],
            'prevention': ['Regular skin checks', 'Sun protection', 'Monitor for changes', 'Annual dermatology visit'],
            'when_to_see_doctor': 'See dermatologist if appearance changes or concerns arise'
        }
    }
    
    return render_template('disease_info.html', diseases=diseases)

@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    """Generate and download PDF report of analysis results."""
    if not REPORTLAB_AVAILABLE:
        return jsonify({'error': 'PDF generation not available. Please install reportlab: pip install reportlab'}), 400
    
    try:
        data = request.get_json()
        
        # Create PDF in memory
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#667eea'), alignment=1, spaceAfter=20)
        story.append(Paragraph('Skin Disease Analysis Report', title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        date_style = ParagraphStyle('DateStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#666666'))
        story.append(Paragraph(f'<b>Report Generated:</b> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', date_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Analysis Results
        results_title = ParagraphStyle('SectionTitle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#333333'), spaceAfter=12)
        story.append(Paragraph('Analysis Results', results_title))
        
        # Results table
        results_data = [
            ['Disease Detected', data.get('disease', 'Unknown')],
            ['Confidence Score', f"{data.get('confidence', 0)}%"],
            ['Severity Level', data.get('severity', 'Unknown')],
            ['Detection Method', data.get('method', 'Image Analysis')],
        ]
        
        results_table = Table(results_data, colWidths=[2*inch, 3*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#ffffff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Description
        story.append(Paragraph('Description', results_title))
        story.append(Paragraph(data.get('description', 'No description available'), styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Symptoms
        story.append(Paragraph('Symptoms', results_title))
        symptoms = data.get('symptoms', [])
        for symptom in symptoms:
            story.append(Paragraph(f'• {symptom}', styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        story.append(Paragraph('Recommendations', results_title))
        recommendations = data.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f'{i}. {rec}', styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Urgency
        urgency_style = ParagraphStyle('UrgencyStyle', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#e74c3c'), bold=True)
        story.append(Paragraph(f'⚠️ Urgency: {data.get("urgency", "Non-urgent")}', urgency_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#999999'))
        story.append(Paragraph(
            '<b>Disclaimer:</b> This analysis is for informational purposes only and should not be considered a medical diagnosis. '
            'Please consult with a qualified healthcare professional for proper diagnosis and treatment.',
            disclaimer_style
        ))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        
        filename = f"skin_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': f'PDF generation error: {str(e)}'}), 500

# ===========================
# ERROR HANDLERS
# ===========================
def download_all():
    """Download all 500 test images as a ZIP file."""
    try:
        uploads_path = Path(app.config['UPLOAD_FOLDER'])
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Get all image files
            image_files = sorted([
                f for f in uploads_path.glob('*.*')
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
            ])
            
            for image_path in image_files:
                # Add file to ZIP with its name
                zip_file.write(image_path, arcname=image_path.name)
        
        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='skin_disease_test_images.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===========================
# TIER 1 ADVANCED FEATURE ENDPOINTS
# ===========================

@app.route('/api/confidence-analysis', methods=['POST'])
def confidence_analysis():
    """Enhanced confidence scoring with reasoning."""
    try:
        data = request.get_json()
        confidence = data.get('confidence', 50)
        
        confidence_category = get_confidence_category(confidence)
        
        return jsonify({
            'success': True,
            'confidence': confidence,
            'analysis': confidence_category
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/abcde-score', methods=['POST'])
def abcde_score_endpoint():
    """Calculate ABCDE Melanoma Risk Score."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            abcde_result = calculate_abcde_score(filepath)
            
            # Clean up
            os.remove(filepath)
            
            if abcde_result:
                return jsonify({
                    'success': True,
                    'abcde_score': abcde_result
                })
            else:
                return jsonify({'error': 'Could not calculate ABCDE score'}), 400
        
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhance-image', methods=['POST'])
def enhance_image_endpoint():
    """Enhance image for better analysis."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        brightness = request.form.get('brightness', 1.0, type=float)
        contrast = request.form.get('contrast', 1.0, type=float)
        color_balance = request.form.get('color_balance', 1.0, type=float)
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = enhance_image(filepath, brightness, contrast, color_balance)
        
        if result['success']:
            with open(result['enhanced_path'], 'rb') as f:
                enhanced_data = base64.b64encode(f.read()).decode()
            
            os.remove(filepath)
            os.remove(result['enhanced_path'])
            
            return jsonify({
                'success': True,
                'enhanced_image': enhanced_data,
                'settings': result['settings']
            })
        else:
            os.remove(filepath)
            return jsonify({'error': result.get('error', 'Enhancement failed')}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/similar-cases', methods=['POST'])
def similar_cases_endpoint():
    """Get similar historical cases."""
    try:
        data = request.get_json()
        disease = data.get('disease', 'Acne')
        confidence = data.get('confidence', 50)
        
        cases = get_similar_cases(disease, confidence)
        
        return jsonify({
            'success': True,
            'similar_cases': cases
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ===========================
# ERROR HANDLERS
# ===========================
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size: 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 error."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 error."""
    return jsonify({'error': 'Internal server error'}), 500

# ===========================
# MAIN
# ===========================
if __name__ == '__main__':
    print("Starting Skin Disease Classification App...")
    print(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
    print(f"Model available: {os.path.exists('models/skin_model.h5')}")
    print("Recognition method: Image Analysis (Color-based classification)")
    app.run(debug=True, host='0.0.0.0', port=8000)
