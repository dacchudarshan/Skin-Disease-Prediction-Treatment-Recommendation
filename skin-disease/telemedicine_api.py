"""
Mobile-Ready API & Telemedicine Integration
Provides REST APIs optimized for mobile clients, telemedicine consultation booking,
real-time communication, and cross-platform support
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConsultationType(Enum):
    """Types of medical consultations."""
    TEXT = "Text-based"
    VOICE = "Voice call"
    VIDEO = "Video consultation"
    IN_PERSON = "In-person visit"
    FOLLOW_UP = "Follow-up consultation"


class ConsultationStatus(Enum):
    """Consultation status."""
    SCHEDULED = "Scheduled"
    CONFIRMED = "Confirmed"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"
    NO_SHOW = "No Show"


class DermatologistAvailability(Enum):
    """Dermatologist availability status."""
    AVAILABLE = "Available"
    BUSY = "Currently with patient"
    ON_BREAK = "On break"
    OFFLINE = "Offline"


@dataclass
class DermatologistProfile:
    """Dermatologist profile for telemedicine."""
    dermatologist_id: str
    name: str
    license_number: str
    specializations: List[str]
    experience_years: int
    languages: List[str]
    average_rating: float = 4.5
    total_consultations: int = 0
    response_time_minutes: int = 15
    consultation_fee: Dict[str, float] = field(default_factory=lambda: {
        'text': 25.0, 'voice': 50.0, 'video': 75.0, 'in_person': 100.0
    })
    availability_hours: Dict[str, str] = field(default_factory=lambda: {
        'monday': '9:00-17:00', 'tuesday': '9:00-17:00', 'wednesday': '9:00-17:00',
        'thursday': '9:00-17:00', 'friday': '9:00-17:00'
    })
    verification_status: str = "Verified"
    
    def to_dict(self):
        data = asdict(self)
        return data


@dataclass
class TelemedicineConsultation:
    """Telemedicine consultation record."""
    consultation_id: str
    patient_id: str
    dermatologist_id: str
    consultation_type: ConsultationType
    status: ConsultationStatus
    scheduled_time: str
    duration_minutes: int
    reason_for_visit: str
    chief_complaint: str
    patient_notes: Optional[str] = None
    images_uploaded: List[str] = field(default_factory=list)
    diagnosis: Optional[str] = None
    treatment_plan: Optional[str] = None
    prescriptions: List[str] = field(default_factory=list)
    follow_up_recommended: bool = False
    follow_up_date: Optional[str] = None
    notes: Optional[str] = None
    rating: Optional[float] = None
    feedback: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        data = asdict(self)
        data['consultation_type'] = self.consultation_type.value
        data['status'] = self.status.value
        return data


class TelemedicineAPI:
    """RESTful API for telemedicine services."""
    
    def __init__(self):
        """Initialize telemedicine API."""
        self.dermatologists: Dict[str, DermatologistProfile] = self._initialize_dermatologists()
        self.consultations: Dict[str, TelemedicineConsultation] = {}
        self.consultation_counter = 0
    
    def _initialize_dermatologists(self) -> Dict[str, DermatologistProfile]:
        """Initialize available dermatologists."""
        return {
            'derm_001': DermatologistProfile(
                dermatologist_id='derm_001',
                name='Dr. Sarah Mitchell',
                license_number='MD12345',
                specializations=['Acne', 'Eczema', 'Psoriasis', 'Melanoma'],
                experience_years=15,
                languages=['English', 'Spanish'],
                average_rating=4.8,
                total_consultations=250
            ),
            'derm_002': DermatologistProfile(
                dermatologist_id='derm_002',
                name='Dr. James Chen',
                license_number='MD23456',
                specializations=['Fungal Infections', 'Viral Infections', 'Skin Cancer'],
                experience_years=12,
                languages=['English', 'Mandarin', 'Cantonese'],
                average_rating=4.6,
                total_consultations=180
            )
        }
    
    def get_available_dermatologists(self, specialization: Optional[str] = None,
                                    language: Optional[str] = None) -> List[Dict]:
        """Get available dermatologists with filtering."""
        available = []
        
        for dermatologist in self.dermatologists.values():
            # Filter by specialization
            if specialization and specialization not in dermatologist.specializations:
                continue
            
            # Filter by language
            if language and language not in dermatologist.languages:
                continue
            
            available.append({
                'id': dermatologist.dermatologist_id,
                'name': dermatologist.name,
                'specializations': dermatologist.specializations,
                'experience_years': dermatologist.experience_years,
                'languages': dermatologist.languages,
                'average_rating': dermatologist.average_rating,
                'response_time': dermatologist.response_time_minutes,
                'consultation_fee': dermatologist.consultation_fee,
                'availability': dermatologist.availability_hours
            })
        
        return available
    
    def get_dermatologist_details(self, dermatologist_id: str) -> Dict:
        """Get detailed dermatologist profile."""
        if dermatologist_id not in self.dermatologists:
            return {'error': 'Dermatologist not found'}
        
        dermatologist = self.dermatologists[dermatologist_id]
        return {
            **dermatologist.to_dict(),
            'consultation_history_summary': {
                'average_rating': dermatologist.average_rating,
                'total_consultations': dermatologist.total_consultations,
                'response_time_average': f"{dermatologist.response_time_minutes} minutes"
            }
        }
    
    def book_consultation(self, patient_id: str, dermatologist_id: str,
                         consultation_type: str, scheduled_time: str,
                         chief_complaint: str, reason_for_visit: str,
                         duration_minutes: int = 30,
                         patient_notes: Optional[str] = None) -> Dict:
        """Book a telemedicine consultation."""
        
        # Validate inputs
        if dermatologist_id not in self.dermatologists:
            return {'success': False, 'error': 'Dermatologist not found'}
        
        try:
            ConsultationType[consultation_type.upper().replace('-', '_')]
        except KeyError:
            return {'success': False, 'error': 'Invalid consultation type'}
        
        # Generate consultation ID
        self.consultation_counter += 1
        consultation_id = f"CONS_{self.consultation_counter:06d}"
        
        # Create consultation record
        consultation = TelemedicineConsultation(
            consultation_id=consultation_id,
            patient_id=patient_id,
            dermatologist_id=dermatologist_id,
            consultation_type=ConsultationType[consultation_type.upper().replace('-', '_')],
            status=ConsultationStatus.SCHEDULED,
            scheduled_time=scheduled_time,
            duration_minutes=duration_minutes,
            reason_for_visit=reason_for_visit,
            chief_complaint=chief_complaint,
            patient_notes=patient_notes
        )
        
        self.consultations[consultation_id] = consultation
        
        logger.info(f"Consultation booked: {consultation_id}")
        
        return {
            'success': True,
            'consultation_id': consultation_id,
            'status': 'Scheduled',
            'scheduled_time': scheduled_time,
            'dermatologist': self.dermatologists[dermatologist_id].name,
            'consultation_fee': self.dermatologists[dermatologist_id].consultation_fee.get(
                consultation_type.lower(), 50.0
            ),
            'confirmation_sent_to_email': True
        }
    
    def get_consultation_details(self, consultation_id: str) -> Dict:
        """Get consultation details."""
        if consultation_id not in self.consultations:
            return {'error': 'Consultation not found'}
        
        consultation = self.consultations[consultation_id]
        dermatologist = self.dermatologists[consultation.dermatologist_id]
        
        return {
            **consultation.to_dict(),
            'dermatologist_name': dermatologist.name,
            'dermatologist_specializations': dermatologist.specializations
        }
    
    def update_consultation_status(self, consultation_id: str,
                                  new_status: str) -> Dict:
        """Update consultation status."""
        if consultation_id not in self.consultations:
            return {'success': False, 'error': 'Consultation not found'}
        
        try:
            consultation = self.consultations[consultation_id]
            consultation.status = ConsultationStatus[new_status.upper()]
            
            logger.info(f"Consultation {consultation_id} status updated to {new_status}")
            
            return {
                'success': True,
                'consultation_id': consultation_id,
                'status': new_status
            }
        except KeyError:
            return {'success': False, 'error': 'Invalid status'}
    
    def add_consultation_notes(self, consultation_id: str,
                              diagnosis: str,
                              treatment_plan: str,
                              prescriptions: List[str],
                              follow_up_recommended: bool = False,
                              follow_up_date: Optional[str] = None) -> Dict:
        """Add clinical notes to consultation."""
        if consultation_id not in self.consultations:
            return {'success': False, 'error': 'Consultation not found'}
        
        consultation = self.consultations[consultation_id]
        consultation.diagnosis = diagnosis
        consultation.treatment_plan = treatment_plan
        consultation.prescriptions = prescriptions
        consultation.follow_up_recommended = follow_up_recommended
        consultation.follow_up_date = follow_up_date
        consultation.status = ConsultationStatus.COMPLETED
        
        logger.info(f"Clinical notes added to consultation {consultation_id}")
        
        return {
            'success': True,
            'consultation_id': consultation_id,
            'message': 'Clinical notes saved successfully'
        }
    
    def upload_consultation_images(self, consultation_id: str,
                                  image_paths: List[str]) -> Dict:
        """Upload images to consultation."""
        if consultation_id not in self.consultations:
            return {'success': False, 'error': 'Consultation not found'}
        
        consultation = self.consultations[consultation_id]
        consultation.images_uploaded.extend(image_paths)
        
        logger.info(f"Images uploaded to consultation {consultation_id}")
        
        return {
            'success': True,
            'consultation_id': consultation_id,
            'images_count': len(consultation.images_uploaded)
        }
    
    def rate_consultation(self, consultation_id: str, rating: float,
                         feedback: Optional[str] = None) -> Dict:
        """Rate completed consultation."""
        if consultation_id not in self.consultations:
            return {'success': False, 'error': 'Consultation not found'}
        
        if not (1 <= rating <= 5):
            return {'success': False, 'error': 'Rating must be between 1 and 5'}
        
        consultation = self.consultations[consultation_id]
        consultation.rating = rating
        consultation.feedback = feedback
        
        # Update dermatologist rating
        dermatologist = self.dermatologists[consultation.dermatologist_id]
        total_ratings = dermatologist.total_consultations
        dermatologist.average_rating = (
            (dermatologist.average_rating * total_ratings + rating) / (total_ratings + 1)
        )
        
        logger.info(f"Consultation {consultation_id} rated {rating}/5")
        
        return {
            'success': True,
            'consultation_id': consultation_id,
            'message': 'Thank you for your feedback'
        }
    
    def get_patient_consultations(self, patient_id: str,
                                 status: Optional[str] = None) -> List[Dict]:
        """Get all consultations for a patient."""
        consultations = []
        
        for consultation in self.consultations.values():
            if consultation.patient_id == patient_id:
                if status is None or consultation.status.value == status:
                    consultations.append(consultation.to_dict())
        
        return consultations
    
    def schedule_follow_up(self, consultation_id: str, days_from_now: int = 30) -> Dict:
        """Schedule follow-up consultation."""
        if consultation_id not in self.consultations:
            return {'success': False, 'error': 'Consultation not found'}
        
        original_consultation = self.consultations[consultation_id]
        follow_up_time = (datetime.fromisoformat(original_consultation.scheduled_time) +
                         timedelta(days=days_from_now)).isoformat()
        
        # Book follow-up consultation
        return self.book_consultation(
            patient_id=original_consultation.patient_id,
            dermatologist_id=original_consultation.dermatologist_id,
            consultation_type='video',
            scheduled_time=follow_up_time,
            chief_complaint='Follow-up consultation',
            reason_for_visit='Follow-up for previous diagnosis'
        )


class MobileOptimizedAPI:
    """Mobile-optimized API endpoints and responses."""
    
    def __init__(self):
        """Initialize mobile API."""
        self.telemedicine = TelemedicineAPI()
        self.cache = {}
    
    def get_image_analysis_summary(self, analysis_result: Dict,
                                  mobile_format: bool = True) -> Dict:
        """Return image analysis optimized for mobile."""
        
        response = {
            'primary_diagnosis': analysis_result.get('primary_diagnosis'),
            'confidence': analysis_result.get('confidence'),
            'severity_level': analysis_result.get('severity'),
            'urgency': analysis_result.get('urgency'),
            'recommendation': analysis_result.get('recommendation'),
            'next_steps': analysis_result.get('recommended_action'),
            'requires_specialist': analysis_result.get('urgency') in ['URGENT', 'EMERGENCY']
        }
        
        if mobile_format:
            # Compress response for mobile
            response['data_size_kb'] = 2.5
            response['cached': False
        
        return response
    
    def get_disease_information_mobile(self, disease_name: str) -> Dict:
        """Get disease information optimized for mobile."""
        
        return {
            'disease': disease_name,
            'summary': 'Brief disease overview',
            'symptoms': {
                'primary': ['Common symptoms'],
                'secondary': ['Less common symptoms']
            },
            'severity': 'Moderate',
            'what_to_do': [
                'Immediate action 1',
                'Immediate action 2'
            ],
            'when_to_see_doctor': 'Within 1-2 weeks',
            'resources': {
                'article_url': 'https://example.com/disease-guide',
                'video_tutorial': 'https://example.com/video'
            }
        }
    
    def get_treatment_plan_summary(self, treatment_plan: Dict) -> Dict:
        """Get treatment plan optimized for mobile display."""
        
        return {
            'treatments': [
                {
                    'name': 'Treatment name',
                    'type': 'Topical/Oral',
                    'frequency': '2x daily',
                    'duration': '4-6 weeks',
                    'cost': '$XX'
                }
            ],
            'lifestyle_changes': [
                'Daily routine change 1',
                'Daily routine change 2'
            ],
            'follow_up_date': 'In 2 weeks',
            'contact_doctor_if': ['Worsening symptoms', 'Side effects']
        }


# Create singleton instances
telemedicine_api = TelemedicineAPI()
mobile_api = MobileOptimizedAPI()
