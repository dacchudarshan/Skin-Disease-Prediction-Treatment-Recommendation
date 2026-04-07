"""
Comprehensive Healthcare Knowledge Base & Treatment Recommendation Engine
Provides evidence-based treatment guidelines, healthcare recommendations,
and personalized treatment plans for skin diseases
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Disease severity classification."""
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"
    CRITICAL = "Critical"


class UrgencyLevel(Enum):
    """Medical urgency levels."""
    NON_URGENT = "Non-urgent"
    ROUTINE = "Routine (within 1-2 weeks)"
    URGENT = "Urgent (within 24-48 hours)"
    EMERGENCY = "Emergency (immediate)"


@dataclass
class Medication:
    """Medication information with safety data."""
    name: str
    category: str  # OTC, Prescription, Natural
    type: str  # Topical, Oral, Injectable
    active_ingredient: str
    strength: str
    indications: List[str]
    contraindications: List[str]
    side_effects: List[str]
    drug_interactions: List[str]
    pregnancy_safe: bool
    age_restrictions: Dict[str, int]  # age_group: min_age
    cost_range: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TreatmentOption:
    """Individual treatment option."""
    name: str
    type: str  # Topical, Oral, Procedural, Lifestyle
    description: str
    efficacy: float  # 0-100%
    duration: str  # How long to see results
    cost_range: str
    side_effects: List[str]
    contraindications: List[str]
    apply_frequency: Optional[str] = None
    duration_of_treatment: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DiseaseInfo:
    """Comprehensive disease information."""
    disease_name: str
    icd10_code: str
    severity: SeverityLevel
    urgency: UrgencyLevel
    description: str
    pathophysiology: str
    epidemiology: Dict[str, str]  # prevalence, age_of_onset, etc.
    symptoms: List[str]
    risk_factors: List[str]
    complications: List[str]
    
    def to_dict(self):
        data = asdict(self)
        data['severity'] = self.severity.value
        data['urgency'] = self.urgency.value
        return data


class TreatmentRecommendationEngine:
    """Evidence-based treatment recommendation system."""
    
    def __init__(self):
        """Initialize the knowledge base."""
        self.disease_knowledge_base = self._build_disease_database()
        self.medication_database = self._build_medication_database()
        self.treatment_protocols = self._build_treatment_protocols()
        self.lifestyle_recommendations = self._build_lifestyle_recommendations()
        self.preventive_measures = self._build_preventive_measures()
    
    def _build_disease_database(self) -> Dict:
        """Build comprehensive disease knowledge base."""
        return {
            'Acne/Rosacea': {
                'icd10_code': 'L70',
                'severity': SeverityLevel.MODERATE,
                'urgency': UrgencyLevel.NON_URGENT,
                'description': 'Chronic inflammatory skin condition characterized by comedones, papules, pustules, and nodules',
                'pathophysiology': 'Caused by follicular hyperkeratinization, increased sebum production, bacterial colonization, and inflammation',
                'epidemiology': {
                    'prevalence': '9.4% of global population',
                    'most_common_age': '12-24 years',
                    'affects_males': '85%',
                    'affects_females': '85%'
                },
                'symptoms': ['Comedones (blackheads/whiteheads)', 'Papules', 'Pustules', 'Nodules', 'Cysts', 'Redness', 'Oily skin'],
                'risk_factors': ['Hormonal changes', 'Genetic predisposition', 'Poor hygiene', 'Certain medications', 'Diet', 'Stress'],
                'complications': ['Acne scars', 'Post-inflammatory hyperpigmentation', 'Psychological distress', 'Cystic acne'],
                'triggers': ['Hormonal fluctuations', 'Dietary factors (high glycemic index)', 'Stress', 'Certain cosmetics']
            },
            'Eczema/Dermatitis': {
                'icd10_code': 'L20',
                'severity': SeverityLevel.MODERATE,
                'urgency': UrgencyLevel.ROUTINE,
                'description': 'Inflammatory skin disorder causing itching, redness, swelling, and cracking',
                'pathophysiology': 'Characterized by disrupted skin barrier, T-cell mediated immunity dysfunction, and increased IgE levels',
                'epidemiology': {
                    'prevalence': '1-3% of adults',
                    'prevalence_children': '15-20%',
                    'onset': 'Often in early childhood'
                },
                'symptoms': ['Intense itching', 'Red inflamed skin', 'Dry sensitive skin', 'Swelling', 'Cracking', 'Weeping', 'Crusting'],
                'risk_factors': ['Family history', 'Atopy', 'Environmental triggers', 'Stress', 'Irritants', 'Allergens'],
                'complications': ['Secondary bacterial infection', 'Lichenification', 'Eczema herpeticum', 'Sleep disturbance'],
                'triggers': ['Irritants (soaps, detergents)', 'Allergens', 'Temperature/humidity changes', 'Stress', 'Sweat']
            },
            'Psoriasis': {
                'icd10_code': 'L40',
                'severity': SeverityLevel.MODERATE,
                'urgency': UrgencyLevel.ROUTINE,
                'description': 'Chronic autoimmune skin condition with thick, scaly, red patches',
                'pathophysiology': 'T-cell mediated autoimmune disorder with keratinocyte hyperproliferation and abnormal differentiation',
                'epidemiology': {
                    'prevalence': '0.5-2% of global population',
                    'age_of_onset': 'Typically 15-35 or 50-60 years',
                    'affects_males_females': 'Equal'
                },
                'symptoms': ['Red scaly patches', 'Silvery scales', 'Burning sensation', 'Itching', 'Joint pain (arthritis)', 'Nail changes'],
                'risk_factors': ['Genetic predisposition', 'Stress', 'Infection (streptococcal)', 'Trauma', 'Certain medications'],
                'complications': ['Psoriatic arthritis', 'Metabolic syndrome', 'Depression', 'Eye involvement'],
                'triggers': ['Stress', 'Infection', 'Skin trauma', 'Certain medications (beta-blockers, lithium)']
            },
            'Fungal Infection': {
                'icd10_code': 'B35-B36',
                'severity': SeverityLevel.MILD,
                'urgency': UrgencyLevel.ROUTINE,
                'description': 'Skin infection caused by dermatophytes, yeasts, or other fungi',
                'pathophysiology': 'Fungal colonization of stratum corneum with inflammatory host response',
                'epidemiology': {
                    'prevalence': '10-15% of infections',
                    'most_common': 'Tinea pedis (athletes foot)'
                },
                'symptoms': ['Red patches', 'Itching', 'Scaling', 'Maceration', 'Burning sensation', 'White patches (candida)'],
                'risk_factors': ['Warm moist environment', 'Poor hygiene', 'Immunosuppression', 'Diabetes', 'Antibiotic use'],
                'complications': ['Secondary bacterial infection', 'Cellulitis', 'Systemic infection (in immunocompromised)'],
                'triggers': ['Moisture', 'Heat', 'Poor hygiene', 'Close contact with infected person']
            },
            'Viral Infection': {
                'icd10_code': 'A90-B34',
                'severity': SeverityLevel.MILD,
                'urgency': UrgencyLevel.NON_URGENT,
                'description': 'Skin manifestation of viral infection (herpes, warts, chickenpox, etc.)',
                'pathophysiology': 'Viral invasion of keratinocytes with host immune response',
                'epidemiology': {
                    'most_common': 'Herpes simplex virus (HSV)',
                    'prevalence_hsv': '67% of global population'
                },
                'symptoms': ['Vesicles (fluid-filled blisters)', 'Pustules', 'Crusting', 'Pain', 'Burning', 'Itching', 'Swelling'],
                'risk_factors': ['Immunosuppression', 'Close contact', 'Stress', 'Fatigue', 'UV exposure'],
                'complications': ['Secondary bacterial infection', 'Disseminated infection', 'Post-herpetic neuralgia'],
                'triggers': ['Stress', 'Immunosuppression', 'UV exposure', 'Fever']
            },
            'Melanoma/Mole (High Risk)': {
                'icd10_code': 'C43',
                'severity': SeverityLevel.CRITICAL,
                'urgency': UrgencyLevel.EMERGENCY,
                'description': 'Most serious form of skin cancer arising from melanocytes',
                'pathophysiology': 'Malignant transformation of melanocytes with uncontrolled proliferation and metastatic potential',
                'epidemiology': {
                    'incidence': '20 per 100,000',
                    '5_year_survival': '92% if caught early'
                },
                'symptoms': ['Asymmetry', 'Irregular border', 'Color variation', 'Diameter > 6mm', 'Evolution/change', 'Itching', 'Bleeding'],
                'risk_factors': ['UV exposure (UVB, UVA)', 'Family history', 'Personal history of melanoma', 'Atypical nevi', 'Skin type (fair)'],
                'complications': ['Regional lymph node metastasis', 'Distant metastasis', 'High mortality'],
                'triggers': ['Sun exposure', 'Artificial UV exposure (tanning beds)']
            }
        }
    
    def _build_medication_database(self) -> Dict:
        """Build comprehensive medication database."""
        return {
            'Topical Treatments': {
                'benzoyl_peroxide': Medication(
                    name='Benzoyl Peroxide',
                    category='OTC',
                    type='Topical',
                    active_ingredient='Benzoyl Peroxide 2.5%-10%',
                    strength='2.5%, 5%, 10%',
                    indications=['Acne', 'Rosacea'],
                    contraindications=['Severe salicylate sensitivity', 'Pregnancy (caution)'],
                    side_effects=['Dryness', 'Irritation', 'Redness', 'Allergic contact dermatitis'],
                    drug_interactions=['Tretinoin', 'Vitamin C (reduces effectiveness)'],
                    pregnancy_safe=False,
                    age_restrictions={'minimum': 12},
                    cost_range='$5-15'
                ),
                'tretinoin': Medication(
                    name='Tretinoin (Retin-A)',
                    category='Prescription',
                    type='Topical',
                    active_ingredient='Tretinoin (Vitamin A derivative)',
                    strength='0.025%, 0.05%, 0.1%',
                    indications=['Acne', 'Photoaging', 'Wrinkles'],
                    contraindications=['Pregnancy', 'Breastfeeding', 'Severe eczema'],
                    side_effects=['Retinization (peeling, dryness)', 'Sun sensitivity', 'Irritation'],
                    drug_interactions=['Benzoyl peroxide', 'Vitamin C', 'AHAs/BHAs'],
                    pregnancy_safe=False,
                    age_restrictions={'minimum': 12},
                    cost_range='$20-100'
                ),
                'hydrocortisone': Medication(
                    name='Hydrocortisone Cream',
                    category='OTC',
                    type='Topical',
                    active_ingredient='Hydrocortisone 0.5%-2.5%',
                    strength='0.5%, 1%, 2.5%',
                    indications=['Eczema', 'Dermatitis', 'Inflammation'],
                    contraindications=['Viral infections', 'Vaccinia', 'Varicella'],
                    side_effects=['Skin atrophy', 'Striae', 'Telangiectasia (with prolonged use)'],
                    drug_interactions=['None significant'],
                    pregnancy_safe=True,
                    age_restrictions={'minimum': 2},
                    cost_range='$5-10'
                ),
                'antifungal_cream': Medication(
                    name='Miconazole/Clotrimazole',
                    category='OTC',
                    type='Topical',
                    active_ingredient='Miconazole 2% or Clotrimazole 1%',
                    strength='1%-2%',
                    indications=['Fungal infections', 'Candida', 'Tinea'],
                    contraindications=['Hypersensitivity'],
                    side_effects=['Irritation', 'Contact dermatitis', 'Burning'],
                    drug_interactions=['None significant'],
                    pregnancy_safe=True,
                    age_restrictions={'minimum': 2},
                    cost_range='$8-15'
                )
            },
            'Oral Medications': {
                'doxycycline': Medication(
                    name='Doxycycline',
                    category='Prescription',
                    type='Oral',
                    active_ingredient='Doxycycline Hyclate',
                    strength='50mg, 100mg',
                    indications=['Acne', 'Rosacea', 'Bacterial infections'],
                    contraindications=['Pregnancy', 'Children < 8 years', 'Myasthenia gravis'],
                    side_effects=['Photosensitivity', 'Nausea', 'Vaginal yeast infection', 'Esophageal ulceration'],
                    drug_interactions=['Antacids', 'Dairy products', 'Iron supplements'],
                    pregnancy_safe=False,
                    age_restrictions={'minimum': 8},
                    cost_range='$10-50'
                ),
                'antihistamine': Medication(
                    name='Cetirizine/Loratadine',
                    category='OTC',
                    type='Oral',
                    active_ingredient='Cetirizine 10mg or Loratadine 10mg',
                    strength='10mg',
                    indications=['Itching', 'Allergic reactions', 'Urticaria'],
                    contraindications=['Severe liver disease'],
                    side_effects=['Drowsiness (cetirizine less)', 'Headache', 'Dry mouth'],
                    drug_interactions=['Alcohol', 'CNS depressants'],
                    pregnancy_safe=True,
                    age_restrictions={'minimum': 2},
                    cost_range='$5-12'
                )
            }
        }
    
    def _build_treatment_protocols(self) -> Dict:
        """Build evidence-based treatment protocols."""
        return {
            'Acne/Rosacea': {
                'mild': [
                    TreatmentOption(
                        name='Daily Cleansing',
                        type='Lifestyle',
                        description='Gentle cleansing twice daily with mild cleanser',
                        efficacy=60,
                        duration='3-4 weeks',
                        cost_range='$5-15',
                        side_effects=[],
                        contraindications=[],
                        apply_frequency='2x daily',
                        duration_of_treatment='Ongoing'
                    ),
                    TreatmentOption(
                        name='Benzoyl Peroxide 2.5%',
                        type='Topical',
                        description='Antimicrobial and keratolytic agent',
                        efficacy=70,
                        duration='2-4 weeks',
                        cost_range='$5-15',
                        side_effects=['Dryness', 'Irritation'],
                        contraindications=['Severe sensitivity'],
                        apply_frequency='1-2x daily',
                        duration_of_treatment='8-12 weeks'
                    )
                ],
                'moderate': [
                    TreatmentOption(
                        name='Retinoid Therapy (Tretinoin 0.025%)',
                        type='Topical',
                        description='Vitamin A derivative for acne reduction and skin renewal',
                        efficacy=85,
                        duration='6-12 weeks',
                        cost_range='$20-100',
                        side_effects=['Retinization', 'Photosensitivity'],
                        contraindications=['Pregnancy', 'Severe sensitivity'],
                        apply_frequency='3-4x weekly initially',
                        duration_of_treatment='12+ weeks'
                    ),
                    TreatmentOption(
                        name='Oral Doxycycline 100mg',
                        type='Oral',
                        description='Antibiotic with anti-inflammatory properties',
                        efficacy=80,
                        duration='4-6 weeks',
                        cost_range='$10-50',
                        side_effects=['Photosensitivity', 'GI upset'],
                        contraindications=['Pregnancy', 'Breastfeeding', 'Age < 8'],
                        apply_frequency='Once daily',
                        duration_of_treatment='3-6 months'
                    )
                ],
                'severe': [
                    TreatmentOption(
                        name='Isotretinoin (Accutane)',
                        type='Oral',
                        description='Only treatment that can cure severe acne',
                        efficacy=95,
                        duration='6 months',
                        cost_range='$4000-8000',
                        side_effects=['Dry skin/lips', 'Photosensitivity', 'Potential teratogenicity'],
                        contraindications=['Pregnancy', 'Breastfeeding'],
                        apply_frequency='Once daily',
                        duration_of_treatment='16-24 weeks'
                    )
                ]
            },
            'Eczema/Dermatitis': {
                'mild': [
                    TreatmentOption(
                        name='Moisturizer (Ceramide-rich)',
                        type='Lifestyle',
                        description='Fragrance-free, hypoallergenic moisturizer',
                        efficacy=75,
                        duration='Immediate',
                        cost_range='$10-30',
                        side_effects=[],
                        contraindications=[],
                        apply_frequency='2-3x daily',
                        duration_of_treatment='Ongoing'
                    ),
                    TreatmentOption(
                        name='Hydrocortisone 1% Cream',
                        type='Topical',
                        description='Mild topical corticosteroid',
                        efficacy=75,
                        duration='2-3 days',
                        cost_range='$5-10',
                        side_effects=['Minimal with short-term use'],
                        contraindications=['Viral infections'],
                        apply_frequency='2x daily',
                        duration_of_treatment='Up to 1 week'
                    )
                ],
                'moderate': [
                    TreatmentOption(
                        name='Topical Corticosteroid (Triamcinolone 0.1%)',
                        type='Topical',
                        description='Mid-potency topical steroid',
                        efficacy=85,
                        duration='3-7 days',
                        cost_range='$10-20',
                        side_effects=['Skin atrophy with prolonged use'],
                        contraindications=['Viral infections', 'Long-term facial use'],
                        apply_frequency='2x daily',
                        duration_of_treatment='1-2 weeks'
                    ),
                    TreatmentOption(
                        name='Topical Calcineurin Inhibitor (Tacrolimus)',
                        type='Topical',
                        description='Non-steroidal anti-inflammatory',
                        efficacy=80,
                        duration='2-4 weeks',
                        cost_range='$50-150',
                        side_effects=['Burning sensation', 'Risk of skin cancer (rare)'],
                        contraindications=['Viral infections', 'Vaccinia'],
                        apply_frequency='2x daily',
                        duration_of_treatment='Ongoing'
                    )
                ]
            },
            'Fungal Infection': {
                'mild': [
                    TreatmentOption(
                        name='Antifungal Cream (Miconazole 2%)',
                        type='Topical',
                        description='Broad-spectrum topical antifungal',
                        efficacy=85,
                        duration='2-4 weeks',
                        cost_range='$8-15',
                        side_effects=['Irritation', 'Contact dermatitis'],
                        contraindications=['Hypersensitivity'],
                        apply_frequency='2-3x daily',
                        duration_of_treatment='2-4 weeks'
                    )
                ],
                'moderate': [
                    TreatmentOption(
                        name='Oral Antifungal (Terbinafine)',
                        type='Oral',
                        description='Systemic antifungal therapy',
                        efficacy=90,
                        duration='4-6 weeks',
                        cost_range='$30-100',
                        side_effects=['Taste changes', 'Hepatotoxicity'],
                        contraindications=['Liver disease', 'Drug interactions'],
                        apply_frequency='Once daily',
                        duration_of_treatment='4-12 weeks'
                    )
                ]
            }
        }
    
    def _build_lifestyle_recommendations(self) -> Dict:
        """Build comprehensive lifestyle modification recommendations."""
        return {
            'general': {
                'hygiene': [
                    'Wash affected areas twice daily with mild cleanser',
                    'Avoid excessive scrubbing or harsh soaps',
                    'Pat dry gently, do not rub',
                    'Keep nails short to prevent scratching',
                    'Maintain good hand hygiene to prevent secondary infection'
                ],
                'skincare': [
                    'Use fragrance-free, hypoallergenic products',
                    'Avoid irritants: alcohol, acetone, astringents',
                    'Moisturize immediately after bathing',
                    'Use lukewarm water, not hot',
                    'Limit sun exposure; use SPF 30+ sunscreen',
                    'Avoid makeup or use non-comedogenic products'
                ],
                'lifestyle': [
                    'Manage stress through meditation, yoga, exercise',
                    'Get 7-9 hours of quality sleep',
                    'Exercise regularly (30 min, 5 days/week)',
                    'Maintain healthy body weight',
                    'Avoid smoking and alcohol',
                    'Avoid touching or picking at skin'
                ]
            },
            'dietary': [
                'Increase antioxidant-rich foods (berries, leafy greens)',
                'Consume omega-3 fatty acids (fish, flaxseed)',
                'Limit high glycemic index foods',
                'Reduce dairy intake if acne-prone',
                'Stay hydrated (8 glasses water/day)',
                'Avoid trigger foods (spicy, fatty, processed)'
            ],
            'sleep': [
                'Maintain consistent sleep schedule (10pm-6am ideal)',
                'Ensure 7-9 hours of quality sleep',
                'Keep bedroom cool (60-67°F)',
                'Use silk pillowcases to reduce friction',
                'Avoid screens 1 hour before bedtime',
                'Sleep on back to reduce face pressure'
            ],
            'stress_management': [
                'Practice meditation (10-20 min daily)',
                'Deep breathing exercises (4-7-8 technique)',
                'Regular physical exercise',
                'Yoga or tai chi',
                'Progressive muscle relaxation',
                'Journaling or creative outlets',
                'Professional counseling if needed'
            ]
        }
    
    def _build_preventive_measures(self) -> Dict:
        """Build preventive care guidelines."""
        return {
            'sun_protection': [
                'Use broad-spectrum SPF 30+ daily',
                'Reapply sunscreen every 2 hours',
                'Avoid peak sun hours (10am-4pm)',
                'Wear protective clothing, hats, sunglasses',
                'Avoid tanning beds and sun lamps'
            ],
            'skin_barrier': [
                'Maintain skin pH balance',
                'Use ceramide-rich moisturizers',
                'Avoid over-washing',
                'Limit hot showers',
                'Use humidifier in dry climates'
            ],
            'infection_prevention': [
                'Keep skin clean and dry',
                'Avoid sharing personal items (towels, razors)',
                'Trim nails regularly',
                'Avoid scratching or skin trauma',
                'Use protective footwear in public areas'
            ],
            'monitoring': [
                'Monthly self-skin checks using ABCDE rule',
                'Annual dermatology check-up',
                'Photograph suspicious lesions for comparison',
                'Track disease progression',
                'Document trigger identification'
            ]
        }
    
    def get_disease_information(self, disease_name: str) -> Optional[Dict]:
        """Get comprehensive disease information."""
        if disease_name not in self.disease_knowledge_base:
            return None
        
        disease_data = self.disease_knowledge_base[disease_name]
        return {
            'disease_name': disease_name,
            'icd10_code': disease_data.get('icd10_code'),
            'severity': disease_data['severity'].value,
            'urgency': disease_data['urgency'].value,
            'description': disease_data['description'],
            'pathophysiology': disease_data['pathophysiology'],
            'epidemiology': disease_data['epidemiology'],
            'symptoms': disease_data['symptoms'],
            'risk_factors': disease_data['risk_factors'],
            'complications': disease_data['complications'],
            'triggers': disease_data.get('triggers', [])
        }
    
    def get_treatment_plan(self, disease_name: str, severity: str) -> List[Dict]:
        """Get personalized treatment plan based on disease and severity."""
        if disease_name not in self.treatment_protocols:
            return []
        
        severity_level = severity.lower()
        if severity_level not in self.treatment_protocols[disease_name]:
            return []
        
        treatments = self.treatment_protocols[disease_name][severity_level]
        return [t.to_dict() for t in treatments]
    
    def get_lifestyle_recommendations(self, recommendation_type: str = 'general') -> List[str]:
        """Get lifestyle modification recommendations."""
        if recommendation_type in self.lifestyle_recommendations:
            if isinstance(self.lifestyle_recommendations[recommendation_type], dict):
                # Flatten nested structure
                result = []
                for key, values in self.lifestyle_recommendations[recommendation_type].items():
                    result.extend(values)
                return result
            else:
                return self.lifestyle_recommendations[recommendation_type]
        
        return []
    
    def get_preventive_care_plan(self) -> Dict:
        """Get comprehensive preventive care guidelines."""
        return self.preventive_measures
    
    def get_medication_info(self, medication_name: str) -> Optional[Dict]:
        """Get detailed medication information."""
        for category, meds in self.medication_database.items():
            for med_key, medication in meds.items():
                if med_key == medication_name.lower().replace(' ', '_'):
                    return medication.to_dict()
        
        return None
    
    def check_drug_interactions(self, medications: List[str]) -> List[Dict]:
        """Check for drug interactions between medications."""
        interactions = []
        
        for i, med1_name in enumerate(medications):
            for med2_name in medications[i+1:]:
                med1 = self.get_medication_info(med1_name)
                med2 = self.get_medication_info(med2_name)
                
                if med1 and med2:
                    if med2_name in med1.get('drug_interactions', []):
                        interactions.append({
                            'drug1': med1_name,
                            'drug2': med2_name,
                            'severity': 'High',
                            'recommendation': 'Consult pharmacist or physician'
                        })
        
        return interactions
    
    def generate_personalized_treatment_plan(self, 
                                            disease_name: str,
                                            severity: str,
                                            user_profile: Dict) -> Dict:
        """Generate personalized treatment plan based on disease and user profile."""
        plan = {
            'disease': disease_name,
            'severity': severity,
            'created_at': datetime.now().isoformat(),
            'disease_info': self.get_disease_information(disease_name),
            'treatment_options': self.get_treatment_plan(disease_name, severity),
            'lifestyle_modifications': self.get_lifestyle_recommendations(),
            'preventive_measures': self.get_preventive_care_plan(),
            'warnings': [],
            'follow_up': None
        }
        
        # Add age-specific recommendations
        age = user_profile.get('age', 30)
        if age < 18:
            plan['age_specific'] = 'Note: Consider pediatric-specific treatments'
        elif age > 60:
            plan['age_specific'] = 'Note: Consider age-related skin changes and comorbidities'
        
        # Add gender-specific recommendations
        gender = user_profile.get('gender')
        if gender == 'Female' and disease_name in ['Acne/Rosacea']:
            plan['gender_specific'] = 'Consider hormonal factors; use contraceptives if treating acne'
        
        # Check for allergies and contraindications
        allergies = user_profile.get('allergies', [])
        if allergies:
            for treatment in plan['treatment_options']:
                for allergy in allergies:
                    if allergy in treatment.get('contraindications', []):
                        plan['warnings'].append(f"⚠️ {treatment['name']} contraindicated due to {allergy} allergy")
        
        # Determine follow-up timeline
        disease_info = self.get_disease_information(disease_name)
        if disease_info:
            urgency = disease_info['urgency']
            if urgency == 'EMERGENCY':
                plan['follow_up'] = 'Immediate (same day)'
            elif urgency == 'URGENT':
                plan['follow_up'] = 'Within 24-48 hours'
            elif urgency == 'ROUTINE':
                plan['follow_up'] = 'Within 1-2 weeks'
            else:
                plan['follow_up'] = 'As needed or if worsening'
        
        return plan


# Create singleton instance
treatment_engine = TreatmentRecommendationEngine()
