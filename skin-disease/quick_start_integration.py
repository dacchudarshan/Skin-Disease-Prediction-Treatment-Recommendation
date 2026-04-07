"""
Quick Start Integration Guide
Demonstrates how to use all advanced modules in your Flask application
"""

from flask import Flask, request, jsonify, session
from deep_learning_models import create_model, EnsembleDeepLearning
from treatment_recommendations import treatment_engine
from user_management import user_management
from security_compliance import (
    security_manager, audit_logger, rbac, compliance_manager,
    UserRole, DataClassification
)
from analytics_reporting import analytics_engine
from telemedicine_api import telemedicine_api
from localization_accessibility import localization_engine, accessibility_manager
import numpy as np
from PIL import Image
import io


# =======================
# EXAMPLE 1: DEEP LEARNING
# =======================

def example_deep_learning_usage():
    """Example: Using advanced deep learning models."""
    
    # Create individual models
    efficientnet = create_model('efficientnet', num_classes=7)
    resnet = create_model('resnet50', num_classes=7)
    
    # Create ensemble for best accuracy
    ensemble = EnsembleDeepLearning([efficientnet, resnet], weights=[0.6, 0.4])
    
    # Load sample image
    # In production, this would come from file upload
    sample_image = np.random.rand(224, 224, 3) * 255
    
    # Get ensemble prediction
    result = ensemble.predict(sample_image)
    
    print("Ensemble Predictions:")
    print(result['ensemble_predictions'])
    print("Top 3 predictions:", result['top_3'])
    
    # Extract features for visualization or research
    features = efficientnet.extract_features(sample_image)
    print(f"Extracted features shape: {features.shape}")
    
    return result


# ==============================
# EXAMPLE 2: TREATMENT PLANNING
# ==============================

def example_treatment_planning():
    """Example: Generate personalized treatment plan."""
    
    # Get disease information
    disease_info = treatment_engine.get_disease_information('Acne/Rosacea')
    print(f"\nDisease: {disease_info['disease_name']}")
    print(f"Severity: {disease_info['severity']}")
    print(f"Symptoms: {', '.join(disease_info['symptoms'][:3])}")
    
    # Get treatment options
    treatments = treatment_engine.get_treatment_plan('Acne/Rosacea', 'moderate')
    print(f"\nRecommended treatments ({len(treatments)} options):")
    for t in treatments:
        print(f"  - {t['name']}: {t['description']}")
    
    # Get lifestyle recommendations
    lifestyle = treatment_engine.get_lifestyle_recommendations('general')
    print(f"\nLifestyle recommendations:")
    for rec in lifestyle[:3]:
        print(f"  - {rec}")
    
    # Check medication interactions
    meds = ['benzoyl_peroxide', 'tretinoin']
    interactions = treatment_engine.check_drug_interactions(meds)
    print(f"\nDrug interactions found: {len(interactions)}")
    
    # Generate full personalized plan
    user_profile = {
        'age': 25,
        'gender': 'Female',
        'allergies': []
    }
    plan = treatment_engine.generate_personalized_treatment_plan(
        'Acne/Rosacea', 'moderate', user_profile
    )
    print(f"\nPersonalized plan created with {len(plan['treatment_options'])} options")
    
    return plan


# =========================
# EXAMPLE 3: USER MANAGEMENT
# =========================

def example_user_management():
    """Example: Manage user profiles and medical history."""
    
    # Create user account
    result = user_management.create_user(
        email='patient@example.com',
        password='SecurePass123!',
        name='John Doe',
        date_of_birth='1990-01-15',
        gender='Male',
        skin_type='TYPE_III',
        location='New York, USA',
        occupation='Software Engineer'
    )
    
    user_id = result['user_id']
    print(f"User created: {user_id}")
    
    # Add medical history
    user_management.add_medical_history(user_id, {
        'allergies': ['Penicillin', 'Pollen'],
        'medications': ['Aspirin 100mg daily'],
        'comorbidities': ['Hypertension']
    })
    print("Medical history added")
    
    # Record disease occurrence
    result = user_management.record_disease_occurrence(
        user_id, 'Acne/Rosacea', 'moderate',
        ['Red patches', 'Papules', 'Burning sensation']
    )
    print(f"Disease recorded: {result['disease_id']}")
    
    # Update disease progression
    for i in range(5):
        user_management.update_disease_progression(
            user_id, 'Acne/Rosacea',
            severity=f'moderate-{i}',
            symptoms=['Improving symptoms'],
            notes=f'Week {i+1} assessment'
        )
    print("Disease progression tracked over 5 weeks")
    
    # Get progression summary
    summary = user_management.get_disease_progression_summary(user_id, 'Acne/Rosacea')
    print(f"Disease duration: {summary['disease_duration_days']} days")
    print(f"Current severity: {summary['current_severity']}")
    
    # Get personalized recommendations
    recommendations = user_management.get_personalized_recommendations(user_id)
    print(f"Risk factors: {recommendations['risk_factors']}")
    
    return user_id


# ===========================
# EXAMPLE 4: SECURITY & AUDIT
# ===========================

def example_security_compliance():
    """Example: Security features and compliance."""
    
    # Authenticate user
    result = user_management.authenticate_user('patient@example.com', 'SecurePass123!')
    user_id = result['user_id']
    session_token = result['session_token']
    print(f"User authenticated: {user_id}")
    print(f"Session token: {session_token[:20]}...")
    
    # Setup RBAC
    rbac.assign_role(user_id, UserRole.PATIENT)
    print(f"Role assigned: PATIENT")
    
    # Check permissions
    can_view = rbac.has_permission(user_id, 'view_own_data')
    print(f"Can view own data: {can_view}")
    
    # Log access for audit
    audit_entry = audit_logger.log_access(
        user_id=user_id,
        action='PATIENT_RECORD_ACCESS',
        resource='medical_record',
        details={'record_type': 'disease_history'},
        ip_address='192.168.1.1',
        data_classification=DataClassification.RESTRICTED
    )
    print(f"Audit logged: {audit_entry.action}")
    
    # Get compliance report
    hipaa_report = compliance_manager.get_compliance_report(
        compliance_manager.standards[0]  # HIPAA
    )
    print(f"HIPAA Status: {hipaa_report['status']}")
    print(f"Requirements: {len(hipaa_report['requirements'])} items")
    
    # Get security status
    security_status = security_manager.get_security_status()
    print(f"\nSecurity Status:")
    print(f"  Encryption: {security_status['encryption']}")
    print(f"  Audit logging: {security_status['audit_logging']}")
    print(f"  MFA available: {security_status['mfa_available']}")
    
    return user_id


# ================================
# EXAMPLE 5: ANALYTICS & REPORTING
# ================================

def example_analytics_reporting():
    """Example: Generate reports and analytics."""
    
    # Sample patient data
    patient_data = {
        'patient_id': 'PAT001',
        'disease': 'Psoriasis',
        'severity_history': [30, 35, 40, 45, 50],
        'duration_days': 180,
        'age': 45,
        'compliance_rate': 0.8,
        'comorbidities': ['Diabetes'],
        'symptoms': ['Red patches', 'Scaling', 'Itching'],
        'current_treatments': ['Topical steroid', 'Moisturizer']
    }
    
    # Generate patient report
    report = analytics_engine.generate_patient_report(patient_data)
    print(f"\nPatient Report Generated:")
    print(f"  Disease: {report.disease_summary['diagnosis']}")
    print(f"  Current severity: {report.disease_summary['current_severity']}")
    print(f"  Risk level: {report.risk_assessment['level']}")
    print(f"  Follow-up: {report.follow_up_schedule['clinical_visits']}")
    
    # Predict disease progression
    progression = analytics_engine.prognostic_predictor.predict_disease_progression(
        patient_data, 'Psoriasis', prediction_period_days=90
    )
    print(f"\n90-Day Progression Prediction:")
    print(f"  Current severity: {progression['current_severity']}")
    print(f"  Projected severity: {progression['projected_severity_at_end']}")
    print(f"  Risk level: {progression['risk_level']}")
    print(f"  Trend: {progression['progression_trend']}")
    
    # Calculate risk score
    risk = analytics_engine.risk_stratification.calculate_disease_risk_score(patient_data)
    print(f"\nRisk Assessment:")
    print(f"  Risk score: {risk['risk_score']}/100")
    print(f"  Risk level: {risk['risk_level']}")
    print(f"  Key factors: {', '.join(risk['risk_factors'][:2])}")
    
    return report


# ==================================
# EXAMPLE 6: TELEMEDICINE INTEGRATION
# ==================================

def example_telemedicine():
    """Example: Book and manage telemedicine consultations."""
    
    # Get available dermatologists
    dermatologists = telemedicine_api.get_available_dermatologists(
        specialization='Acne'
    )
    print(f"\nAvailable dermatologists: {len(dermatologists)}")
    for derm in dermatologists[:2]:
        print(f"  - {derm['name']} ({derm['average_rating']}⭐)")
    
    # Book consultation
    consultation = telemedicine_api.book_consultation(
        patient_id='PAT001',
        dermatologist_id='derm_001',
        consultation_type='VIDEO',
        scheduled_time='2025-01-15T14:00:00',
        chief_complaint='Persistent acne',
        reason_for_visit='Treatment optimization',
        patient_notes='Tried benzoyl peroxide, some improvement'
    )
    
    if consultation['success']:
        print(f"\nConsultation booked: {consultation['consultation_id']}")
        print(f"  Fee: ${consultation['consultation_fee']}")
        print(f"  Scheduled: {consultation['scheduled_time']}")
        
        consultation_id = consultation['consultation_id']
        
        # Update consultation status
        telemedicine_api.update_consultation_status(
            consultation_id, 'CONFIRMED'
        )
        
        # Add clinical notes after consultation
        telemedicine_api.add_consultation_notes(
            consultation_id,
            diagnosis='Moderate acne with rosacea',
            treatment_plan='Topical retinoid + oral antibiotic',
            prescriptions=['Tretinoin 0.05%', 'Doxycycline 100mg'],
            follow_up_recommended=True,
            follow_up_date='2025-02-15'
        )
        print("Clinical notes added")


# ===================================
# EXAMPLE 7: LOCALIZATION & TRANSLATION
# ===================================

def example_localization():
    """Example: Multi-language support."""
    from localization_accessibility import Language
    
    # Get available languages
    languages = localization_engine.supported_languages
    print(f"\nSupported languages: {len(languages)}")
    
    # Switch to Spanish
    localization_engine.set_language(Language.SPANISH)
    spanish_strings = localization_engine.get_all_strings()
    print(f"\nSpanish translations loaded: {len(spanish_strings)} strings")
    
    # Get specific strings
    print(f"  Home: {localization_engine.get_string('nav_home')}")
    print(f"  Upload: {localization_engine.get_string('btn_upload')}")
    
    # Enable accessibility features
    from localization_accessibility import AccessibilityFeature
    accessibility_manager.enable_feature(AccessibilityFeature.TEXT_TO_SPEECH)
    accessibility_manager.enable_feature(AccessibilityFeature.HIGH_CONTRAST)
    
    features = accessibility_manager.get_accessibility_settings()
    print(f"\nAccessibility features enabled: {len(features['enabled_features'])}")
    
    # Generate TTS
    audio = text_to_speech.text_to_speech(
        "Skin disease analysis results",
        language=Language.SPANISH
    )
    print(f"Audio generated: {audio['audio_file']}")


# ==================
# EXAMPLE 8: FLASK INTEGRATION
# ==================

def create_flask_app_with_advanced_features():
    """Create Flask app with all advanced features integrated."""
    
    app = Flask(__name__)
    app.secret_key = 'your-secret-key-change-in-production'
    
    # Initialize models
    ensemble_model = EnsembleDeepLearning([
        create_model('efficientnet'),
        create_model('resnet50')
    ])
    
    @app.route('/api/v2/analyze', methods=['POST'])
    def advanced_analysis():
        """Advanced disease analysis endpoint."""
        try:
            # Get image
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided'}), 400
            
            file = request.files['image']
            img = Image.open(io.BytesIO(file.read()))
            img_array = np.array(img.resize((224, 224))) / 255.0
            
            # Get prediction
            prediction = ensemble_model.predict(img_array)
            
            # Get treatment plan
            disease = list(prediction['ensemble_predictions'].keys())[0]
            treatment_plan = treatment_engine.get_treatment_plan(disease, 'moderate')
            
            # Log access
            audit_logger.log_access(
                user_id=session.get('user_id', 'anonymous'),
                action='IMAGE_ANALYSIS',
                resource='medical_image',
                details={'disease': disease},
                ip_address=request.remote_addr,
                data_classification=DataClassification.RESTRICTED
            )
            
            return jsonify({
                'disease': disease,
                'confidence': prediction['ensemble_predictions'][disease],
                'treatment_options': treatment_plan
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/v2/user/profile', methods=['GET'])
    def get_profile():
        """Get user profile with recommendations."""
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401
        
        profile = user_management.get_user_profile(user_id)
        recommendations = user_management.get_personalized_recommendations(user_id)
        
        return jsonify({
            'profile': profile,
            'recommendations': recommendations
        })
    
    @app.route('/api/v2/compliance/status', methods=['GET'])
    def compliance_status():
        """Get compliance status."""
        status = {
            'hipaa': compliance_manager.get_compliance_report(
                compliance_manager.standards[0]
            ),
            'security': security_manager.get_security_status()
        }
        return jsonify(status)
    
    return app


# ================
# MAIN EXECUTION
# ================

if __name__ == '__main__':
    print("=" * 60)
    print("SKIN DISEASE DETECTION - ADVANCED FEATURES DEMO")
    print("=" * 60)
    
    print("\n[1] Testing Deep Learning Models...")
    example_deep_learning_usage()
    
    print("\n[2] Testing Treatment Planning...")
    example_treatment_planning()
    
    print("\n[3] Testing User Management...")
    user_id = example_user_management()
    
    print("\n[4] Testing Security & Compliance...")
    example_security_compliance()
    
    print("\n[5] Testing Analytics & Reporting...")
    example_analytics_reporting()
    
    print("\n[6] Testing Telemedicine...")
    example_telemedicine()
    
    print("\n[7] Testing Localization...")
    example_localization()
    
    print("\n[8] Creating Flask App with Advanced Features...")
    app = create_flask_app_with_advanced_features()
    print("✓ Flask app created successfully")
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
