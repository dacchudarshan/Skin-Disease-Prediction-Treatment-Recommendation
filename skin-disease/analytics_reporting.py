"""
Advanced Analytics & Reporting System
Provides comprehensive analysis, prognostic predictions, risk stratification,
epidemiological trending, and detailed patient/clinical reports
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk stratification levels."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class TrendDirection(Enum):
    """Disease trend direction."""
    IMPROVING = "Improving"
    STABLE = "Stable"
    WORSENING = "Worsening"
    RAPID_DECLINE = "Rapidly Worsening"


@dataclass
class PatientReport:
    """Comprehensive patient report."""
    patient_id: str
    report_date: str
    disease_summary: Dict
    current_status: Dict
    severity_trend: List[Tuple[str, float]]
    risk_assessment: Dict
    recommended_actions: List[str]
    follow_up_schedule: Dict
    medication_review: Dict
    lifestyle_recommendations: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ClinicalMetrics:
    """Clinical performance metrics."""
    model_accuracy: float
    sensitivity: float  # True positive rate
    specificity: float  # True negative rate
    precision: float
    f1_score: float
    auc_roc: float
    confusion_matrix: Dict
    disease_specific_metrics: Dict
    confidence_interval: Tuple[float, float]


class PrognosticPredictor:
    """Predicts disease progression and outcomes."""
    
    def __init__(self):
        """Initialize prognostic predictor."""
        self.progression_models = {}
        self.outcome_predictors = {}
    
    def predict_disease_progression(self, patient_data: Dict, 
                                   disease_name: str,
                                   prediction_period_days: int = 90) -> Dict:
        """Predict disease progression over specified period."""
        
        # Extract relevant patient factors
        age = patient_data.get('age', 30)
        severity_history = patient_data.get('severity_history', [])
        compliance_rate = patient_data.get('compliance_rate', 0.7)
        comorbidities = len(patient_data.get('comorbidities', []))
        
        # Calculate progression probability
        if severity_history:
            # Simplified progression model
            recent_severity = float(severity_history[-1]) / 100 if severity_history else 0.5
            
            # Factors affecting progression
            age_factor = 1.0 + (age - 40) * 0.01  # Age influence
            compliance_factor = 1.0 - (compliance_rate * 0.5)  # Compliance effect
            comorbidity_factor = 1.0 + (comorbidities * 0.15)  # Comorbidity burden
            
            # Combined progression probability
            progression_prob = min(0.95, recent_severity * age_factor * compliance_factor * comorbidity_factor)
        else:
            progression_prob = 0.3
        
        # Determine progression likelihood
        if progression_prob > 0.7:
            progression_trend = TrendDirection.RAPID_DECLINE
            risk_level = RiskLevel.CRITICAL
        elif progression_prob > 0.5:
            progression_trend = TrendDirection.WORSENING
            risk_level = RiskLevel.HIGH
        elif progression_prob > 0.3:
            progression_trend = TrendDirection.STABLE
            risk_level = RiskLevel.MODERATE
        else:
            progression_trend = TrendDirection.IMPROVING
            risk_level = RiskLevel.LOW
        
        # Predict expected severity at end of period
        days_per_unit = 30  # Update period
        projected_months = prediction_period_days / 30
        expected_change = progression_prob * projected_months * 10  # 10% per month max
        
        if severity_history:
            current_severity = float(severity_history[-1])
        else:
            current_severity = 50
        
        projected_severity = min(100, current_severity + expected_change)
        
        return {
            'disease': disease_name,
            'prediction_period_days': prediction_period_days,
            'current_severity': current_severity,
            'projected_severity_at_end': projected_severity,
            'progression_probability': round(progression_prob, 3),
            'progression_trend': progression_trend.value,
            'risk_level': risk_level.value,
            'confidence': round(0.75 + (len(severity_history) * 0.05), 3),
            'contributing_factors': [
                f'Age factor: {round(age_factor, 2)}x',
                f'Compliance factor: {round(compliance_factor, 2)}x',
                f'Comorbidity factor: {round(comorbidity_factor, 2)}x'
            ],
            'recommended_actions': self._get_progression_actions(progression_trend, risk_level),
            'monitoring_frequency': self._get_monitoring_frequency(risk_level)
        }
    
    def predict_treatment_outcome(self, patient_data: Dict,
                                 treatment: str,
                                 duration_days: int = 90) -> Dict:
        """Predict treatment outcome probability."""
        
        # Treatment efficacy baseline rates
        treatment_efficacy = {
            'topical_steroid': 0.75,
            'oral_antibiotic': 0.80,
            'retinoid': 0.85,
            'antifungal': 0.90,
            'immunosuppressant': 0.70,
            'default': 0.70
        }
        
        baseline_efficacy = treatment_efficacy.get(treatment, treatment_efficacy['default'])
        
        # Adjust based on patient factors
        age = patient_data.get('age', 40)
        compliance_rate = patient_data.get('compliance_rate', 0.7)
        comorbidities = len(patient_data.get('comorbidities', []))
        allergies = len(patient_data.get('allergies', []))
        
        # Efficacy modifiers
        age_modifier = 1.0 - ((abs(age - 40) / 100) * 0.1)  # Optimal age range
        compliance_modifier = compliance_rate
        comorbidity_modifier = 1.0 - (comorbidities * 0.1)
        allergy_modifier = 1.0 - (allergies * 0.15)
        
        # Calculate expected outcome
        expected_efficacy = baseline_efficacy * age_modifier * compliance_modifier * comorbidity_modifier * allergy_modifier
        expected_efficacy = max(0.1, min(0.95, expected_efficacy))
        
        # Determine success probability
        if expected_efficacy > 0.8:
            success_probability = "High"
            risk_of_failure = "Low"
        elif expected_efficacy > 0.6:
            success_probability = "Moderate"
            risk_of_failure = "Moderate"
        else:
            success_probability = "Low"
            risk_of_failure = "High"
        
        return {
            'treatment': treatment,
            'duration_days': duration_days,
            'baseline_efficacy': round(baseline_efficacy, 3),
            'expected_efficacy': round(expected_efficacy, 3),
            'success_probability': success_probability,
            'risk_of_failure': risk_of_failure,
            'time_to_improvement_days': int(duration_days * (1 - expected_efficacy) + 7),
            'confidence': round(0.70 + (0.3 * min(1.0, compliance_rate)), 3),
            'risk_factors_for_failure': self._get_failure_risk_factors(
                age, compliance_rate, comorbidities, allergies
            ),
            'alternative_treatments': ['alternative_1', 'alternative_2']
        }
    
    def _get_progression_actions(self, trend: TrendDirection, risk: RiskLevel) -> List[str]:
        """Get recommended actions based on progression trend."""
        actions = []
        
        if risk == RiskLevel.CRITICAL:
            actions.append('Urgent specialist consultation required')
            actions.append('Consider hospitalization if systemic involvement')
        elif risk == RiskLevel.HIGH:
            actions.append('Schedule dermatologist appointment within 48 hours')
            actions.append('Intensify current treatment regimen')
        elif risk == RiskLevel.MODERATE:
            actions.append('Schedule appointment within 1-2 weeks')
            actions.append('Review compliance with current treatment')
        
        return actions
    
    def _get_monitoring_frequency(self, risk: RiskLevel) -> Dict:
        """Get recommended monitoring frequency based on risk."""
        frequency = {
            RiskLevel.CRITICAL: {'clinical_visits': 'Weekly', 'imaging': '2x weekly'},
            RiskLevel.HIGH: {'clinical_visits': 'Bi-weekly', 'imaging': 'Weekly'},
            RiskLevel.MODERATE: {'clinical_visits': 'Monthly', 'imaging': 'Monthly'},
            RiskLevel.LOW: {'clinical_visits': 'Every 3 months', 'imaging': 'Every 3 months'}
        }
        return frequency[risk]
    
    def _get_failure_risk_factors(self, age: int, compliance: float,
                                 comorbidities: int, allergies: int) -> List[str]:
        """Identify risk factors for treatment failure."""
        factors = []
        
        if age > 60:
            factors.append('Advanced age may affect medication response')
        if compliance < 0.6:
            factors.append('Low compliance rate significantly impacts outcome')
        if comorbidities > 2:
            factors.append('Multiple comorbidities may complicate treatment')
        if allergies > 0:
            factors.append('Allergies may limit treatment options')
        
        return factors


class RiskStratificationEngine:
    """Stratifies patients into risk categories."""
    
    def __init__(self):
        """Initialize risk stratification."""
        self.risk_thresholds = {
            RiskLevel.LOW: (0, 30),
            RiskLevel.MODERATE: (30, 60),
            RiskLevel.HIGH: (60, 80),
            RiskLevel.CRITICAL: (80, 100)
        }
    
    def calculate_disease_risk_score(self, patient_data: Dict) -> Dict:
        """Calculate comprehensive disease risk score."""
        
        score = 0
        risk_factors = []
        
        # Age factor (0-20 points)
        age = patient_data.get('age', 40)
        if age < 18:
            score += 5
            risk_factors.append('Adolescent age (hormonal factors)')
        elif age > 60:
            score += 15
            risk_factors.append('Advanced age (higher complication risk)')
        
        # Severity factor (0-25 points)
        severity_history = patient_data.get('severity_history', [])
        if severity_history:
            current_severity = float(severity_history[-1]) / 100
            score += int(current_severity * 25)
            if current_severity > 0.7:
                risk_factors.append('High current severity')
        
        # Progression rate (0-20 points)
        if len(severity_history) > 1:
            progression_rate = (float(severity_history[-1]) - float(severity_history[0])) / max(len(severity_history) - 1, 1)
            if progression_rate > 5:
                score += 20
                risk_factors.append('Rapid disease progression')
            elif progression_rate > 2:
                score += 10
                risk_factors.append('Moderate disease progression')
        
        # Comorbidities (0-15 points)
        comorbidities = len(patient_data.get('comorbidities', []))
        score += min(15, comorbidities * 5)
        if comorbidities > 0:
            risk_factors.append(f'{comorbidities} comorbidities present')
        
        # Compliance (0-20 points)
        compliance = patient_data.get('compliance_rate', 0.7)
        if compliance < 0.5:
            score += 20
            risk_factors.append('Poor treatment compliance')
        elif compliance < 0.7:
            score += 10
            risk_factors.append('Moderate compliance issues')
        
        # Determine risk level
        risk_level = RiskLevel.LOW
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= score <= max_score:
                risk_level = level
                break
        
        return {
            'risk_score': min(100, score),
            'risk_level': risk_level.value,
            'risk_factors': risk_factors,
            'score_breakdown': {
                'age': min(20, int(abs(age - 40) / 2)),
                'severity': len(severity_history) * 5,
                'progression': 5 if len(severity_history) > 3 else 0,
                'comorbidities': min(15, comorbidities * 5),
                'compliance': int((1 - compliance) * 20)
            },
            'recommendations': self._get_risk_recommendations(risk_level)
        }
    
    def _get_risk_recommendations(self, risk_level: RiskLevel) -> List[str]:
        """Get recommendations based on risk level."""
        recommendations = {
            RiskLevel.LOW: [
                'Continue current management',
                'Annual check-ups',
                'Maintain preventive care'
            ],
            RiskLevel.MODERATE: [
                'Quarterly monitoring',
                'Intensify preventive care',
                'Consider prophylactic treatment'
            ],
            RiskLevel.HIGH: [
                'Bi-monthly specialist visits',
                'Advanced diagnostic imaging',
                'Aggressive treatment protocol'
            ],
            RiskLevel.CRITICAL: [
                'Intensive specialist care',
                'Possible hospitalization',
                'Immediate intervention required'
            ]
        }
        return recommendations[risk_level]


class AnalyticsEngine:
    """Advanced analytics and epidemiological trending."""
    
    def __init__(self):
        """Initialize analytics engine."""
        self.prognostic_predictor = PrognosticPredictor()
        self.risk_stratification = RiskStratificationEngine()
        self.patient_data = []
        self.epidemiological_data = defaultdict(list)
    
    def generate_patient_report(self, patient_data: Dict) -> PatientReport:
        """Generate comprehensive patient report."""
        
        patient_id = patient_data['patient_id']
        disease = patient_data.get('disease', 'Unknown')
        severity_history = patient_data.get('severity_history', [])
        
        # Disease summary
        disease_summary = {
            'diagnosis': disease,
            'onset_date': patient_data.get('onset_date'),
            'duration_days': patient_data.get('duration_days', 0),
            'current_severity': severity_history[-1] if severity_history else 50
        }
        
        # Current status
        current_status = {
            'symptoms': patient_data.get('symptoms', []),
            'active_treatments': patient_data.get('current_treatments', []),
            'recent_changes': 'Improving' if len(severity_history) > 1 and severity_history[-1] < severity_history[-2] else 'Stable'
        }
        
        # Severity trend
        severity_trend = [(f"Day {i*7}", s) for i, s in enumerate(severity_history[::7])]
        
        # Risk assessment
        risk_data = self.risk_stratification.calculate_disease_risk_score(patient_data)
        risk_assessment = {
            'score': risk_data['risk_score'],
            'level': risk_data['risk_level'],
            'factors': risk_data['risk_factors']
        }
        
        # Prognostic prediction
        progression_pred = self.prognostic_predictor.predict_disease_progression(patient_data, disease)
        
        # Recommended actions
        recommended_actions = progression_pred['recommended_actions']
        
        # Follow-up schedule
        monitoring = progression_pred['monitoring_frequency']
        follow_up_schedule = {
            'clinical_visits': monitoring['clinical_visits'],
            'imaging': monitoring['imaging'],
            'next_appointment': (datetime.now() + timedelta(days=14)).isoformat()
        }
        
        # Medication review
        medication_review = {
            'current_medications': patient_data.get('current_treatments', []),
            'compliance': patient_data.get('compliance_rate', 0.7),
            'side_effects': patient_data.get('side_effects', [])
        }
        
        # Lifestyle recommendations
        lifestyle_recommendations = [
            'Maintain consistent sleep schedule (7-9 hours)',
            'Manage stress through meditation/exercise',
            'Follow dietary guidelines for condition',
            'Avoid identified triggers'
        ]
        
        return PatientReport(
            patient_id=patient_id,
            report_date=datetime.now().isoformat(),
            disease_summary=disease_summary,
            current_status=current_status,
            severity_trend=severity_trend,
            risk_assessment=risk_assessment,
            recommended_actions=recommended_actions,
            follow_up_schedule=follow_up_schedule,
            medication_review=medication_review,
            lifestyle_recommendations=lifestyle_recommendations
        )
    
    def analyze_epidemiological_trends(self, population_data: List[Dict]) -> Dict:
        """Analyze epidemiological trends in population."""
        
        if not population_data:
            return {}
        
        diseases = Counter([d.get('disease') for d in population_data])
        age_groups = Counter([d.get('age_group') for d in population_data])
        severities = [d.get('severity', 50) for d in population_data if isinstance(d.get('severity'), (int, float))]
        
        return {
            'total_patients_analyzed': len(population_data),
            'disease_distribution': dict(diseases.most_common()),
            'age_group_distribution': dict(age_groups.most_common()),
            'average_severity': round(np.mean(severities), 2) if severities else 0,
            'severity_range': [min(severities), max(severities)] if severities else [0, 0],
            'trending_diseases': [d[0] for d in diseases.most_common(3)],
            'high_risk_age_groups': [a[0] for a in age_groups.most_common(2)],
            'prevalence_rates': {disease: round(count / len(population_data) * 100, 2) 
                                for disease, count in diseases.items()},
            'generated_at': datetime.now().isoformat()
        }
    
    def calculate_clinical_metrics(self, predictions: List[Dict],
                                  actual_values: List[Dict]) -> ClinicalMetrics:
        """Calculate clinical performance metrics."""
        
        # Simplified metric calculation
        if len(predictions) != len(actual_values):
            return None
        
        # TP, TN, FP, FN calculation
        tp = sum(1 for p, a in zip(predictions, actual_values) 
                if p.get('prediction') == a.get('actual') and p.get('prediction') == True)
        tn = sum(1 for p, a in zip(predictions, actual_values) 
                if p.get('prediction') == a.get('actual') and p.get('prediction') == False)
        fp = sum(1 for p, a in zip(predictions, actual_values) 
                if p.get('prediction') == True and a.get('actual') == False)
        fn = sum(1 for p, a in zip(predictions, actual_values) 
                if p.get('prediction') == False and a.get('actual') == True)
        
        # Calculate metrics
        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        return ClinicalMetrics(
            model_accuracy=round(accuracy, 3),
            sensitivity=round(sensitivity, 3),
            specificity=round(specificity, 3),
            precision=round(precision, 3),
            f1_score=round(f1, 3),
            auc_roc=0.92,  # Placeholder
            confusion_matrix={'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            disease_specific_metrics={},
            confidence_interval=(round(accuracy - 0.05, 3), round(accuracy + 0.05, 3))
        )


# Create singleton instance
analytics_engine = AnalyticsEngine()
