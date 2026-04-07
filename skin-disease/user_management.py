"""
User Management & Personalization System
Manages user profiles, medical history, disease progression tracking,
and personalized recommendations
"""

import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SkinType(Enum):
    """Fitzpatrick skin classification."""
    TYPE_I = "Type I (Very Fair)"
    TYPE_II = "Type II (Fair)"
    TYPE_III = "Type III (Medium)"
    TYPE_IV = "Type IV (Olive)"
    TYPE_V = "Type V (Brown)"
    TYPE_VI = "Type VI (Dark)"


class AgeGroup(Enum):
    """Age grouping for recommendations."""
    PEDIATRIC = "Pediatric (0-12)"
    ADOLESCENT = "Adolescent (13-19)"
    ADULT = "Adult (20-49)"
    SENIOR = "Senior (50+)"


@dataclass
class MedicalHistory:
    """User's medical history record."""
    allergies: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    previous_diagnoses: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)
    comorbidities: List[str] = field(default_factory=list)
    current_treatments: List[str] = field(default_factory=list)
    pregnancy_status: Optional[str] = None
    immunocompromised: bool = False
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DiseaseProgression:
    """Track individual disease progression over time."""
    disease_name: str
    initial_date: str
    last_observation_date: str
    severity_history: List[tuple] = field(default_factory=list)  # (date, severity)
    symptoms_history: List[tuple] = field(default_factory=list)  # (date, symptoms)
    treatment_history: List[tuple] = field(default_factory=list)  # (date, treatment)
    progression_notes: List[str] = field(default_factory=list)
    remission_status: Optional[str] = None
    
    def to_dict(self):
        data = asdict(self)
        data['severity_history'] = self.severity_history
        data['symptoms_history'] = self.symptoms_history
        data['treatment_history'] = self.treatment_history
        return data


@dataclass
class UserProfile:
    """Comprehensive user profile for personalized recommendations."""
    user_id: str
    email: str
    name: str
    date_of_birth: str
    gender: str
    skin_type: SkinType
    location: Optional[str] = None
    occupation: Optional[str] = None
    
    # Medical information
    medical_history: MedicalHistory = field(default_factory=MedicalHistory)
    
    # Disease tracking
    disease_progression: Dict[str, DiseaseProgression] = field(default_factory=dict)
    
    # Preferences
    language: str = "English"
    notification_preferences: Dict = field(default_factory=dict)
    
    # Activity tracking
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: Optional[str] = None
    total_sessions: int = 0
    
    def to_dict(self):
        data = {
            'user_id': self.user_id,
            'email': self.email,
            'name': self.name,
            'date_of_birth': self.date_of_birth,
            'gender': self.gender,
            'skin_type': self.skin_type.value,
            'location': self.location,
            'occupation': self.occupation,
            'medical_history': self.medical_history.to_dict(),
            'disease_progression': {k: v.to_dict() for k, v in self.disease_progression.items()},
            'language': self.language,
            'notification_preferences': self.notification_preferences,
            'created_at': self.created_at,
            'last_login': self.last_login,
            'total_sessions': self.total_sessions
        }
        return data
    
    def get_age(self) -> int:
        """Calculate user age."""
        dob = datetime.fromisoformat(self.date_of_birth)
        today = datetime.now()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    
    def get_age_group(self) -> AgeGroup:
        """Get age group classification."""
        age = self.get_age()
        if age <= 12:
            return AgeGroup.PEDIATRIC
        elif age <= 19:
            return AgeGroup.ADOLESCENT
        elif age <= 49:
            return AgeGroup.ADULT
        else:
            return AgeGroup.SENIOR


class UserManagementSystem:
    """Manages user accounts, authentication, and personalization."""
    
    def __init__(self):
        """Initialize user management system."""
        self.users: Dict[str, UserProfile] = {}
        self.user_sessions: Dict[str, Dict] = {}
    
    def create_user(self, email: str, password: str, name: str, 
                   date_of_birth: str, gender: str, skin_type: str,
                   **kwargs) -> Dict:
        """Create new user account."""
        # Check if email already exists
        if any(user.email == email for user in self.users.values()):
            return {'success': False, 'error': 'Email already registered'}
        
        # Generate user ID
        user_id = self._generate_user_id(email)
        
        # Hash password
        password_hash = self._hash_password(password)
        
        try:
            skin_type_enum = SkinType[f"TYPE_{skin_type.upper()}"]
        except KeyError:
            return {'success': False, 'error': 'Invalid skin type'}
        
        # Create user profile
        user_profile = UserProfile(
            user_id=user_id,
            email=email,
            name=name,
            date_of_birth=date_of_birth,
            gender=gender,
            skin_type=skin_type_enum,
            **kwargs
        )
        
        # Store user (in production, use database)
        self.users[user_id] = user_profile
        
        # Store password hash securely (in production, use proper storage)
        self._store_password_hash(user_id, password_hash)
        
        logger.info(f"User created: {email}")
        return {'success': True, 'user_id': user_id, 'message': 'Account created successfully'}
    
    def authenticate_user(self, email: str, password: str) -> Dict:
        """Authenticate user with email and password."""
        # Find user by email
        user = next((u for u in self.users.values() if u.email == email), None)
        
        if not user:
            return {'success': False, 'error': 'User not found'}
        
        # Verify password
        if not self._verify_password(user.user_id, password):
            return {'success': False, 'error': 'Invalid password'}
        
        # Create session
        session_token = self._create_session(user.user_id)
        
        # Update last login
        user.last_login = datetime.now().isoformat()
        user.total_sessions += 1
        
        logger.info(f"User authenticated: {email}")
        return {
            'success': True,
            'user_id': user.user_id,
            'session_token': session_token,
            'user_profile': user.to_dict()
        }
    
    def update_user_profile(self, user_id: str, updates: Dict) -> Dict:
        """Update user profile information."""
        if user_id not in self.users:
            return {'success': False, 'error': 'User not found'}
        
        user = self.users[user_id]
        
        # Allowed fields to update
        allowed_fields = ['name', 'location', 'occupation', 'language', 'notification_preferences']
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(user, field, value)
        
        logger.info(f"User profile updated: {user_id}")
        return {'success': True, 'message': 'Profile updated successfully'}
    
    def add_medical_history(self, user_id: str, history_data: Dict) -> Dict:
        """Add or update medical history."""
        if user_id not in self.users:
            return {'success': False, 'error': 'User not found'}
        
        user = self.users[user_id]
        
        for field, value in history_data.items():
            if hasattr(user.medical_history, field):
                if isinstance(value, list):
                    getattr(user.medical_history, field).extend(value)
                else:
                    setattr(user.medical_history, field, value)
        
        user.medical_history.last_updated = datetime.now().isoformat()
        
        logger.info(f"Medical history updated: {user_id}")
        return {'success': True, 'message': 'Medical history updated'}
    
    def record_disease_occurrence(self, user_id: str, disease_name: str, 
                                 severity: str, symptoms: List[str]) -> Dict:
        """Record initial occurrence of a disease."""
        if user_id not in self.users:
            return {'success': False, 'error': 'User not found'}
        
        user = self.users[user_id]
        
        if disease_name in user.disease_progression:
            return {'success': False, 'error': 'Disease already recorded for this user'}
        
        # Create disease progression record
        progression = DiseaseProgression(
            disease_name=disease_name,
            initial_date=datetime.now().isoformat(),
            last_observation_date=datetime.now().isoformat(),
            severity_history=[(datetime.now().isoformat(), severity)],
            symptoms_history=[(datetime.now().isoformat(), symptoms)]
        )
        
        user.disease_progression[disease_name] = progression
        
        logger.info(f"Disease recorded for user {user_id}: {disease_name}")
        return {
            'success': True,
            'message': f'Recorded {disease_name} for tracking',
            'disease_id': disease_name
        }
    
    def update_disease_progression(self, user_id: str, disease_name: str,
                                  severity: str, symptoms: List[str],
                                  notes: Optional[str] = None) -> Dict:
        """Update disease progression tracking."""
        if user_id not in self.users:
            return {'success': False, 'error': 'User not found'}
        
        user = self.users[user_id]
        
        if disease_name not in user.disease_progression:
            return {'success': False, 'error': 'Disease not found in user records'}
        
        progression = user.disease_progression[disease_name]
        
        # Add to history
        current_time = datetime.now().isoformat()
        progression.severity_history.append((current_time, severity))
        progression.symptoms_history.append((current_time, symptoms))
        progression.last_observation_date = current_time
        
        if notes:
            progression.progression_notes.append(f"[{current_time}] {notes}")
        
        logger.info(f"Disease progression updated for user {user_id}: {disease_name}")
        return {
            'success': True,
            'message': 'Disease progression updated',
            'progression_data': progression.to_dict()
        }
    
    def get_disease_progression_summary(self, user_id: str, disease_name: str) -> Dict:
        """Get summary of disease progression."""
        if user_id not in self.users:
            return {'success': False, 'error': 'User not found'}
        
        user = self.users[user_id]
        
        if disease_name not in user.disease_progression:
            return {'success': False, 'error': 'Disease not found'}
        
        progression = user.disease_progression[disease_name]
        
        # Calculate disease duration
        initial_date = datetime.fromisoformat(progression.initial_date)
        last_date = datetime.fromisoformat(progression.last_observation_date)
        duration = (last_date - initial_date).days
        
        # Get severity trend
        severity_trend = [severity for _, severity in progression.severity_history]
        
        return {
            'success': True,
            'disease_name': disease_name,
            'disease_duration_days': duration,
            'initial_date': progression.initial_date,
            'last_observation_date': progression.last_observation_date,
            'severity_history': severity_trend,
            'current_severity': severity_trend[-1] if severity_trend else None,
            'total_observations': len(progression.severity_history),
            'progression_notes': progression.progression_notes
        }
    
    def get_personalized_recommendations(self, user_id: str) -> Dict:
        """Get personalized recommendations based on user profile and history."""
        if user_id not in self.users:
            return {'success': False, 'error': 'User not found'}
        
        user = self.users[user_id]
        age_group = user.get_age_group()
        
        recommendations = {
            'user_id': user_id,
            'age_group': age_group.value,
            'skin_type': user.skin_type.value,
            'personalized_care_plan': [],
            'risk_factors': [],
            'treatment_contraindications': [],
            'monitoring_recommendations': []
        }
        
        # Age-specific recommendations
        if age_group == AgeGroup.ADOLESCENT:
            recommendations['personalized_care_plan'].append(
                'Hormonal acne management: focus on gentle cleansing and non-comedogenic products'
            )
        elif age_group == AgeGroup.SENIOR:
            recommendations['personalized_care_plan'].append(
                'Age-related skin changes: increased moisturization and anti-aging treatments'
            )
        
        # Skin type specific recommendations
        if user.skin_type.value.startswith('Type I') or user.skin_type.value.startswith('Type II'):
            recommendations['personalized_care_plan'].append(
                'Fair skin: Enhanced sun protection (SPF 50+) and melanoma screening'
            )
            recommendations['risk_factors'].append('High skin cancer risk')
        
        # Medical history based contraindications
        if 'Pregnancy' in user.medical_history.pregnancy_status or 'Pregnant' in str(user.medical_history.pregnancy_status):
            recommendations['treatment_contraindications'].append(
                'Avoid: Retinoids, Isotretinoin, Tetracyclines'
            )
        
        if user.medical_history.immunocompromised:
            recommendations['treatment_contraindications'].append(
                'Monitor closely: increased infection risk'
            )
            recommendations['monitoring_recommendations'].append(
                'Weekly dermatology follow-ups'
            )
        
        # Disease-specific monitoring
        for disease_name in user.disease_progression:
            recommendations['monitoring_recommendations'].append(
                f'{disease_name}: Monthly assessment and photography'
            )
        
        return {'success': True, **recommendations}
    
    def _generate_user_id(self, email: str) -> str:
        """Generate unique user ID."""
        return hashlib.sha256(f"{email}{secrets.token_hex(8)}".encode()).hexdigest()[:16]
    
    def _hash_password(self, password: str) -> str:
        """Hash password securely."""
        salt = secrets.token_hex(32)
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
    
    def _verify_password(self, user_id: str, password: str) -> bool:
        """Verify password (placeholder - implement with proper storage)."""
        # In production, retrieve stored hash from database and compare
        return True
    
    def _store_password_hash(self, user_id: str, password_hash: str):
        """Store password hash securely (placeholder - implement with database)."""
        # In production, use secure database storage
        pass
    
    def _create_session(self, user_id: str) -> str:
        """Create user session token."""
        token = secrets.token_urlsafe(32)
        self.user_sessions[token] = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=7)).isoformat()
        }
        return token
    
    def verify_session(self, token: str) -> Optional[str]:
        """Verify session token and return user_id if valid."""
        if token not in self.user_sessions:
            return None
        
        session = self.user_sessions[token]
        expires_at = datetime.fromisoformat(session['expires_at'])
        
        if datetime.now() > expires_at:
            del self.user_sessions[token]
            return None
        
        return session['user_id']
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile."""
        if user_id in self.users:
            return self.users[user_id].to_dict()
        return None


# Create singleton instance
user_management = UserManagementSystem()
