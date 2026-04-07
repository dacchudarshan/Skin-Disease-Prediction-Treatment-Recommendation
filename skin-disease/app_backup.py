import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image, ImageStat
import base64
import zipfile
import io
from pathlib import Path

# Try to import tensorflow/keras for ML model
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not installed. Using image analysis for classification.")

# ===========================
# CONFIGURATION
# ===========================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
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
    """Predict skin disease from image using ML or image analysis."""
    try:
        # If TensorFlow model is available, use it
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
                'method': 'Deep Learning Model'
            }
        
        # Fallback: Use feature-based classification
        else:
            features = extract_image_features(image_path)
            if features:
                result = classify_skin_disease(features)
                result['method'] = 'Image Analysis'
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

# ===========================
# ROUTES
# ===========================
@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/gallery')
def gallery():
    """View all test images."""
    return render_template('gallery.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'model_exists': os.path.exists('models/skin_model.h5')
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
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
        
        # Load model
        model = load_model()
        
        # Make prediction
        result = predict_disease(filepath, model)
        
        # Add disease information and recommendations
        disease_info = get_disease_information(result['disease'])
        result.update(disease_info)
        
        # Read image for preview
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        result['preview'] = f'data:image/png;base64,{img_data}'
        result['filename'] = filename
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/batch-analysis')
def batch_analysis():
    """Batch image analysis page."""
    return render_template('batch_analysis.html')

@app.route('/statistics')
def statistics():
    """Statistics and analytics dashboard."""
    uploads_path = Path(app.config['UPLOAD_FOLDER'])
    image_files = sorted([
        f for f in uploads_path.glob('*.*')
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    ])
    
    # Categorize images
    categories = {
        'normal': len([f for f in image_files if '01_normal' in f.name]),
        'acne': len([f for f in image_files if '02_acne' in f.name]),
        'eczema': len([f for f in image_files if '03_eczema' in f.name]),
        'psoriasis': len([f for f in image_files if '04_psoriasis' in f.name]),
        'melanoma': len([f for f in image_files if '05_melanoma' in f.name]),
        'nevus': len([f for f in image_files if '06_nevus' in f.name]),
        'dermatitis': len([f for f in image_files if '07_dermatitis' in f.name]),
    }
    
    return jsonify({
        'total_images': len(image_files),
        'categories': categories,
        'category_labels': list(categories.keys()),
        'category_values': list(categories.values())
    })

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

