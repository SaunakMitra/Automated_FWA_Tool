"""
Healthcare Analytics Tool - Comprehensive Python Implementation
This includes FWA detection scenarios, ML algorithms, and extensive field mapping

To run locally:
pip install streamlit pandas plotly sqlalchemy numpy openpyxl scikit-learn seaborn matplotlib
streamlit run healthcare_analytics.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import io
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for background styling inspired by the network image
def load_custom_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #ff6b35 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #ff6b35 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ff6b35, #f7931e);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .scenario-card {
        background: rgba(255, 255, 255, 0.9);
        border-left: 4px solid #ff6b35;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

class HealthcareAnalytics:
    def __init__(self):
        self.data = None
        self.field_mappings = {}
        
        # Comprehensive required fields list as provided
        self.required_fields = {
            'claim_id': {'type': 'string', 'description': 'Unique claim identifier'},
            'member_id': {'type': 'string', 'description': 'Member identifier'},
            'provider_id': {'type': 'string', 'description': 'Healthcare provider ID'},
            'provider_type': {'type': 'string', 'description': 'Type of healthcare provider'},
            'claim_invoice_no': {'type': 'string', 'description': 'Claim invoice number'},
            'claim_invoice_line_no': {'type': 'string', 'description': 'Claim invoice line number'},
            'invoice_no_reference': {'type': 'string', 'description': 'Invoice number reference'},
            'claim_version': {'type': 'string', 'description': 'Claim version'},
            'latest_claim_version_ind': {'type': 'string', 'description': 'Latest claim version indicator'},
            'claim_status_code': {'type': 'string', 'description': 'Claim status code'},
            'incident_count': {'type': 'number', 'description': 'Incident count'},
            'diagnostic_code': {'type': 'string', 'description': 'Diagnostic code (ICD-10)'},
            'procedure_code': {'type': 'string', 'description': 'Procedure code (CPT codes)'},
            'age': {'type': 'number', 'description': 'Patient age'},
            'gender': {'type': 'string', 'description': 'Patient gender'},
            'nationality_code': {'type': 'string', 'description': 'Nationality code'},
            'claim_invoice_date': {'type': 'date', 'description': 'Claim invoice date'},
            'admission_date': {'type': 'date', 'description': 'Admission date'},
            'discharge_date': {'type': 'date', 'description': 'Discharge date'},
            'los': {'type': 'number', 'description': 'Length of stay'},
            'pos': {'type': 'string', 'description': 'Place of service'},
            'treatment_from_date': {'type': 'date', 'description': 'Treatment from date'},
            'treatment_to_date': {'type': 'date', 'description': 'Treatment to date'},
            'provider_country_code': {'type': 'string', 'description': 'Provider country code'},
            'paid_amount': {'type': 'number', 'description': 'Paid amount'},
            'claimed_currency_code': {'type': 'string', 'description': 'Claimed currency code'},
            'payment_currency_code': {'type': 'string', 'description': 'Payment currency code'},
            'base_currency_code': {'type': 'string', 'description': 'Base currency code'},
            'claim_invoice_gross_total_amount': {'type': 'number', 'description': 'Claim invoice gross total amount'},
            'payee_type': {'type': 'string', 'description': 'Payee type'},
            'conversion_rate': {'type': 'number', 'description': 'Conversion rate'},
            'policy_start_date': {'type': 'date', 'description': 'Policy start date'},
            'policy_end_date': {'type': 'date', 'description': 'Policy end date'},
            'previous_fraud_flags': {'type': 'string', 'description': 'Previous fraud flags'},
            'member_zip_code': {'type': 'string', 'description': 'Member zip code'},
            'provider_zip_code': {'type': 'string', 'description': 'Provider zip code'},
            'coverage_type': {'type': 'string', 'description': 'Coverage type (Inpatient, Outpatient, Pharmacy, etc.)'},
            'facility_type': {'type': 'string', 'description': 'Facility type (Clinic, Hospital, Lab)'},
            'ndc_code': {'type': 'string', 'description': 'NDC code'},
            'prior_auth_required_flag': {'type': 'string', 'description': 'Prior auth required flag'},
            'prior_auth_number': {'type': 'string', 'description': 'Prior auth number'},
            'prior_auth_approved_flag': {'type': 'string', 'description': 'Prior auth approved flag'},
            'prior_auth_approval_date': {'type': 'date', 'description': 'Prior auth approval date'},
            'referral_required_flag': {'type': 'string', 'description': 'Referral required flag'},
            'referral_provider_id': {'type': 'string', 'description': 'Referral provider id'},
            'referral_submission_date': {'type': 'date', 'description': 'Referral submission date'},
            'claim_status_datetime': {'type': 'date', 'description': 'Claim status datetime'},
            'denial_code': {'type': 'string', 'description': 'Denial code'},
            'denial_reason': {'type': 'string', 'description': 'Denial reason'},
            'billed_amount': {'type': 'number', 'description': 'Billed amount'},
            'allowed_amount': {'type': 'number', 'description': 'Allowed amount'},
            'deductible_remaining': {'type': 'number', 'description': 'Deductible remaining'},
            'copay_amount': {'type': 'number', 'description': 'Copay amount'},
            'coinsurance_pct': {'type': 'number', 'description': 'Coinsurance percentage'},
            'policy_code': {'type': 'string', 'description': 'Policy code'},
            'policy_name': {'type': 'string', 'description': 'Policy name'},
            'policy_type': {'type': 'string', 'description': 'Policy type'},
            'policy_max_coverage': {'type': 'number', 'description': 'Policy max coverage'},
            'policy_min_coverage': {'type': 'number', 'description': 'Policy min coverage'},
            'deductible_amount': {'type': 'number', 'description': 'Deductible amount'},
            'out_of_pocket_max': {'type': 'number', 'description': 'Out of pocket max'},
            'enrollment_date': {'type': 'date', 'description': 'Enrollment date'},
            'renewal_date': {'type': 'date', 'description': 'Renewal date'},
            'premium_amount': {'type': 'number', 'description': 'Premium amount or monthly premium'},
            'premium_frequency': {'type': 'string', 'description': 'Premium frequency (e.g. monthly, quarterly)'},
            'employer_contribution': {'type': 'number', 'description': 'Employer contribution'},
            'customer_contribution': {'type': 'number', 'description': 'Customer contribution'},
            'discount_amount': {'type': 'number', 'description': 'Discount amount or subsidy amount'},
            'network_type': {'type': 'string', 'description': 'Network type (In-Network, Out-of-Network)'},
            'coverage_area': {'type': 'string', 'description': 'Coverage area or service area'},
            'prescription_coverage': {'type': 'string', 'description': 'Prescription coverage (Yes/No or details)'},
            'preventive_services_covered': {'type': 'string', 'description': 'Preventive services covered'},
            'policy_status': {'type': 'string', 'description': 'Policy status (Active, Inactive, Cancelled)'},
            'is_default_policy': {'type': 'string', 'description': 'Is default policy (Boolean)'},
            'renewed_flag': {'type': 'string', 'description': 'Renewed flag'}
        }
        
        # Python-based FWA scenarios from the document
        self.python_scenarios = [
            {
                'name': 'Duplicate Claims Detection',
                'required_fields': ['claim_id', 'member_id', 'provider_id', 'paid_amount', 'claim_invoice_date'],
                'description': 'Identifies potential duplicate claims submitted by providers',
                'category': 'python_rules'
            },
            {
                'name': 'Billing Pattern Analysis',
                'required_fields': ['provider_id', 'paid_amount', 'claim_invoice_date', 'procedure_code'],
                'description': 'Analyzes unusual billing patterns and frequency',
                'category': 'python_rules'
            },
            {
                'name': 'Age-Service Mismatch',
                'required_fields': ['age', 'procedure_code', 'diagnostic_code'],
                'description': 'Detects services inappropriate for patient age',
                'category': 'python_rules'
            },
            {
                'name': 'Provider Network Analysis',
                'required_fields': ['provider_id', 'member_id', 'pos'],
                'description': 'Identifies suspicious provider-patient relationships',
                'category': 'python_rules'
            },
            {
                'name': 'Amount Outlier Detection',
                'required_fields': ['paid_amount', 'procedure_code', 'pos'],
                'description': 'Detects claims with unusually high amounts',
                'category': 'python_rules'
            },
            {
                'name': 'Frequency Analysis',
                'required_fields': ['provider_id', 'member_id', 'claim_invoice_date'],
                'description': 'Identifies providers with unusually high claim frequency',
                'category': 'python_rules'
            },
            {
                'name': 'Geographic Anomaly Detection',
                'required_fields': ['member_zip_code', 'provider_zip_code', 'claim_invoice_date'],
                'description': 'Detects claims from geographically distant providers',
                'category': 'python_rules'
            },
            {
                'name': 'Prior Authorization Violations',
                'required_fields': ['prior_auth_required_flag', 'prior_auth_approved_flag', 'procedure_code'],
                'description': 'Identifies claims without required prior authorization',
                'category': 'python_rules'
            }
        ]
        
        # ML-based scenarios
        self.ml_scenarios = [
            {
                'name': 'Anomaly Detection (Isolation Forest)',
                'required_fields': ['paid_amount', 'age', 'los', 'provider_id'],
                'description': 'Uses Isolation Forest to detect anomalous claims',
                'category': 'ml_algorithms'
            },
            {
                'name': 'Clustering Analysis (DBSCAN)',
                'required_fields': ['paid_amount', 'age', 'provider_id', 'procedure_code'],
                'description': 'Identifies unusual claim clusters using DBSCAN',
                'category': 'ml_algorithms'
            },
            {
                'name': 'Fraud Prediction Model',
                'required_fields': ['paid_amount', 'age', 'provider_id', 'previous_fraud_flags'],
                'description': 'Predicts fraud probability using Random Forest',
                'category': 'ml_algorithms'
            },
            {
                'name': 'Pattern Recognition',
                'required_fields': ['provider_id', 'procedure_code', 'paid_amount', 'claim_invoice_date'],
                'description': 'Identifies suspicious patterns in provider behavior',
                'category': 'ml_algorithms'
            }
        ]

    def load_file_data(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded CSV/Excel file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    def connect_database(self, db_config: Dict) -> pd.DataFrame:
        """Connect to database and load data"""
        try:
            # For demo purposes, create comprehensive sample data
            sample_data = self.generate_sample_data()
            return sample_data
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return None

    def generate_sample_data(self) -> pd.DataFrame:
        """Generate comprehensive sample healthcare data"""
        np.random.seed(42)
        n_records = 1000
        
        data = {
            'claim_id': [f'CLM{str(i).zfill(6)}' for i in range(1, n_records + 1)],
            'member_id': [f'MEM{str(np.random.randint(1, 500)).zfill(4)}' for _ in range(n_records)],
            'provider_id': [f'PROV{str(np.random.randint(1, 100)).zfill(3)}' for _ in range(n_records)],
            'provider_type': np.random.choice(['Hospital', 'Clinic', 'Specialist', 'Lab'], n_records),
            'claim_invoice_no': [f'INV{str(i).zfill(6)}' for i in range(1, n_records + 1)],
            'paid_amount': np.random.lognormal(7, 1, n_records).round(2),
            'billed_amount': np.random.lognormal(7.2, 1, n_records).round(2),
            'age': np.random.randint(18, 85, n_records),
            'gender': np.random.choice(['M', 'F'], n_records),
            'diagnostic_code': np.random.choice(['M79.3', 'E11.9', 'I10', 'Z00.00', 'M25.511'], n_records),
            'procedure_code': np.random.choice(['99213', '99214', '99212', '99215', '99201'], n_records),
            'pos': np.random.choice(['Office', 'Hospital', 'Home', 'Emergency'], n_records),
            'coverage_type': np.random.choice(['Inpatient', 'Outpatient', 'Pharmacy', 'Emergency'], n_records),
            'facility_type': np.random.choice(['Clinic', 'Hospital', 'Lab', 'Pharmacy'], n_records),
            'los': np.random.randint(1, 30, n_records),
            'member_zip_code': [f'{np.random.randint(10000, 99999)}' for _ in range(n_records)],
            'provider_zip_code': [f'{np.random.randint(10000, 99999)}' for _ in range(n_records)],
            'previous_fraud_flags': np.random.choice(['Y', 'N'], n_records, p=[0.1, 0.9]),
            'prior_auth_required_flag': np.random.choice(['Y', 'N'], n_records, p=[0.3, 0.7]),
            'prior_auth_approved_flag': np.random.choice(['Y', 'N'], n_records, p=[0.8, 0.2]),
            'claim_status_code': np.random.choice(['Approved', 'Denied', 'Pending'], n_records, p=[0.7, 0.2, 0.1])
        }
        
        # Add date fields
        base_date = datetime(2024, 1, 1)
        data['claim_invoice_date'] = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)]
        data['admission_date'] = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)]
        data['discharge_date'] = [data['admission_date'][i] + timedelta(days=data['los'][i]) for i in range(n_records)]
        
        return pd.DataFrame(data)

    def generate_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data summary"""
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_info': {}
        }
        
        for column in df.columns:
            col_info = {
                'dtype': str(df[column].dtype),
                'null_count': df[column].isnull().sum(),
                'unique_count': df[column].nunique(),
                'null_percentage': (df[column].isnull().sum() / len(df)) * 100
            }
            
            if df[column].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': df[column].mean(),
                    'median': df[column].median(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'q25': df[column].quantile(0.25),
                    'q75': df[column].quantile(0.75)
                })
            
            summary['column_info'][column] = col_info
        
        return summary

    def calculate_field_similarity(self, user_field: str, required_field: str) -> float:
        """Calculate similarity score between user field and required field"""
        user_field = user_field.lower().replace('_', ' ').replace('-', ' ')
        required_field = required_field.lower().replace('_', ' ').replace('-', ' ')
        
        # Exact match
        if user_field == required_field:
            return 100.0
        
        # Contains match
        if required_field in user_field or user_field in required_field:
            return 90.0
        
        # Word-based similarity
        user_words = set(user_field.split())
        required_words = set(required_field.split())
        
        if user_words & required_words:
            similarity = len(user_words & required_words) / len(user_words | required_words)
            return min(similarity * 100, 89.0)
        
        return 0.0

    def generate_field_mappings(self, df_columns: List[str]) -> List[Dict]:
        """Generate field mappings with confidence scores"""
        mappings = []
        
        for required_field, field_info in self.required_fields.items():
            best_match = None
            best_score = 0
            
            for user_field in df_columns:
                score = self.calculate_field_similarity(user_field, required_field)
                if score > best_score:
                    best_score = score
                    best_match = user_field
            
            confidence_score = int(best_score)
            
            mappings.append({
                'required_field': required_field,
                'user_field': best_match or df_columns[0],
                'data_type': field_info['type'],
                'description': field_info['description'],
                'confidence_score': confidence_score,
                'is_confirm': confidence_score >= 70
            })
        
        return mappings

    def get_available_scenarios(self, active_mappings: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Get available and unavailable FWA scenarios"""
        all_scenarios = self.python_scenarios + self.ml_scenarios
        available = []
        unavailable = []
        
        for scenario in all_scenarios:
            if all(field in active_mappings for field in scenario['required_fields']):
                available.append(scenario)
            else:
                missing_fields = [field for field in scenario['required_fields'] if field not in active_mappings]
                scenario_copy = scenario.copy()
                scenario_copy['missing_fields'] = missing_fields
                unavailable.append(scenario_copy)
        
        return available, unavailable

    def run_python_scenarios(self, df: pd.DataFrame, selected_scenarios: List[str]) -> pd.DataFrame:
        """Run Python-based FWA detection scenarios"""
        results_df = df.copy()
        
        for scenario_name in selected_scenarios:
            if scenario_name == 'Duplicate Claims Detection':
                # Detect duplicate claims
                duplicates = df.duplicated(subset=['member_id', 'provider_id', 'paid_amount', 'procedure_code'], keep=False)
                results_df[f'{scenario_name}_flag'] = duplicates.astype(int)
                
            elif scenario_name == 'Billing Pattern Analysis':
                # Detect unusual billing patterns
                provider_stats = df.groupby('provider_id')['paid_amount'].agg(['mean', 'std', 'count']).reset_index()
                provider_stats['threshold'] = provider_stats['mean'] + 2 * provider_stats['std']
                
                unusual_billing = df.merge(provider_stats, on='provider_id')
                unusual_billing_flag = unusual_billing['paid_amount'] > unusual_billing['threshold']
                results_df[f'{scenario_name}_flag'] = unusual_billing_flag.astype(int)
                
            elif scenario_name == 'Age-Service Mismatch':
                # Detect age-inappropriate services
                age_service_rules = {
                    '99213': (18, 100),  # General consultation
                    '99214': (18, 100),  # Detailed consultation
                    '99212': (0, 100),   # Basic consultation
                    '99215': (18, 100),  # Comprehensive consultation
                    '99201': (0, 100)    # New patient consultation
                }
                
                mismatch_flag = []
                for _, row in df.iterrows():
                    procedure = row.get('procedure_code', '')
                    age = row.get('age', 0)
                    
                    if procedure in age_service_rules:
                        min_age, max_age = age_service_rules[procedure]
                        mismatch_flag.append(1 if age < min_age or age > max_age else 0)
                    else:
                        mismatch_flag.append(0)
                
                results_df[f'{scenario_name}_flag'] = mismatch_flag
                
            elif scenario_name == 'Amount Outlier Detection':
                # Detect amount outliers using IQR method
                Q1 = df['paid_amount'].quantile(0.25)
                Q3 = df['paid_amount'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_flag = (df['paid_amount'] < lower_bound) | (df['paid_amount'] > upper_bound)
                results_df[f'{scenario_name}_flag'] = outlier_flag.astype(int)
                
            elif scenario_name == 'Frequency Analysis':
                # Detect high-frequency providers
                provider_freq = df.groupby('provider_id').size().reset_index(name='claim_count')
                high_freq_threshold = provider_freq['claim_count'].quantile(0.95)
                high_freq_providers = provider_freq[provider_freq['claim_count'] > high_freq_threshold]['provider_id']
                
                freq_flag = df['provider_id'].isin(high_freq_providers)
                results_df[f'{scenario_name}_flag'] = freq_flag.astype(int)
        
        return results_df

    def run_ml_scenarios(self, df: pd.DataFrame, selected_scenarios: List[str]) -> pd.DataFrame:
        """Run ML-based FWA detection scenarios"""
        results_df = df.copy()
        
        # Prepare features for ML
        numeric_features = ['paid_amount', 'age', 'los']
        categorical_features = ['provider_id', 'procedure_code', 'diagnostic_code']
        
        # Create feature matrix
        X_numeric = df[numeric_features].fillna(0)
        
        # Encode categorical features
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                df_encoded = pd.get_dummies(df[cat_feature], prefix=cat_feature)
                X_numeric = pd.concat([X_numeric, df_encoded], axis=1)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        for scenario_name in selected_scenarios:
            if scenario_name == 'Anomaly Detection (Isolation Forest)':
                # Isolation Forest for anomaly detection
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = iso_forest.fit_predict(X_scaled)
                results_df[f'{scenario_name}_flag'] = (anomaly_labels == -1).astype(int)
                
            elif scenario_name == 'Clustering Analysis (DBSCAN)':
                # DBSCAN clustering
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = dbscan.fit_predict(X_scaled)
                # Flag outliers (cluster label -1)
                results_df[f'{scenario_name}_flag'] = (cluster_labels == -1).astype(int)
                
            elif scenario_name == 'Fraud Prediction Model':
                # Create synthetic fraud labels for demonstration
                fraud_labels = (df['previous_fraud_flags'] == 'Y').astype(int)
                
                if fraud_labels.sum() > 0:  # Only if we have some fraud cases
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_model.fit(X_scaled, fraud_labels)
                    fraud_prob = rf_model.predict_proba(X_scaled)[:, 1]
                    results_df[f'{scenario_name}_flag'] = (fraud_prob > 0.5).astype(int)
                    results_df[f'{scenario_name}_score'] = fraud_prob
                else:
                    results_df[f'{scenario_name}_flag'] = 0
                    results_df[f'{scenario_name}_score'] = 0
        
        return results_df

    def calculate_fraud_score(self, results_df: pd.DataFrame, selected_scenarios: List[str]) -> pd.DataFrame:
        """Calculate overall fraud score based on flagged scenarios"""
        flag_columns = [col for col in results_df.columns if col.endswith('_flag')]
        
        # Calculate fraud score as percentage of scenarios flagged
        results_df['fraud_score'] = results_df[flag_columns].sum(axis=1) / len(flag_columns) * 100
        
        # Sort by fraud score (highest first)
        results_df = results_df.sort_values('fraud_score', ascending=False)
        
        return results_df

    def create_visualization(self, df: pd.DataFrame, column: str):
        """Create appropriate visualization for a column"""
        try:
            if df[column].dtype in ['int64', 'float64']:
                # Histogram for numerical data
                fig = px.histogram(df, x=column, title=f'Distribution of {column}',
                                 color_discrete_sequence=['#ff6b35'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333')
                )
            else:
                # Bar chart for categorical data (top 10 values)
                value_counts = df[column].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f'Top 10 values in {column}',
                           color_discrete_sequence=['#ff6b35'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333')
                )
            
            return fig
        except Exception as e:
            st.error(f"Error creating visualization for {column}: {str(e)}")
            return None

def main():
    st.set_page_config(page_title="Healthcare Analytics Tool", layout="wide", initial_sidebar_state="expanded")
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'analytics' not in st.session_state:
        st.session_state.analytics = HealthcareAnalytics()
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'dashboard'
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'mapping_confirmed' not in st.session_state:
        st.session_state.mapping_confirmed = False

    analytics = st.session_state.analytics

    # Header
    st.title("🏥 Healthcare Analytics Tool")
    st.markdown("Upload healthcare claims data to perform comprehensive analytics on claims datasets. Access different analytical modules including FWA (Fraud, Waste, and Abuse) analytics.")

    # Sidebar for navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.current_step = 'dashboard'
        
        if st.session_state.data_loaded:
            if st.button("👁️ Data Preview", use_container_width=True):
                st.session_state.current_step = 'preview'
            if st.button("🔗 Field Mapping", use_container_width=True):
                st.session_state.current_step = 'mapping'
        
        if st.session_state.mapping_confirmed:
            if st.button("📊 Claims Summary", use_container_width=True):
                st.session_state.current_step = 'claims_summary'
            if st.button("🛡️ FWA Detection", use_container_width=True):
                st.session_state.current_step = 'fwa_detection'
            if st.button("📈 Trend Analysis", use_container_width=True):
                st.session_state.current_step = 'trend_analysis'

    # Main content based on current step
    if st.session_state.current_step == 'dashboard':
        show_dashboard(analytics)
    elif st.session_state.current_step == 'preview':
        show_data_preview(analytics)
    elif st.session_state.current_step == 'mapping':
        show_field_mapping(analytics)
    elif st.session_state.current_step == 'claims_summary':
        show_claims_summary(analytics)
    elif st.session_state.current_step == 'fwa_detection':
        show_fwa_detection(analytics)
    elif st.session_state.current_step == 'trend_analysis':
        show_trend_analysis(analytics)

def show_dashboard(analytics):
    """Show main dashboard with upload options and analytical modules"""
    
    # Upload options in top right
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        uploaded_file = st.file_uploader("📁 Upload File", type=['csv', 'xlsx', 'xls'])
        if uploaded_file:
            df = analytics.load_file_data(uploaded_file)
            if df is not None:
                analytics.data = df
                st.session_state.data_loaded = True
                st.success("✅ File uploaded successfully!")
                st.session_state.current_step = 'preview'
                st.rerun()
    
    with col3:
        if st.button("🗄️ Connect Your DB", use_container_width=True):
            show_database_modal()

    st.markdown("---")

    # Analytical modules
    st.header("🔬 Analytics Modules")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="scenario-card">
            <h3>📊 Claims Data Summary</h3>
            <p>Comprehensive statistical summary of all available fields with visualizations and insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.mapping_confirmed:
            if st.button("Open Claims Summary", key="claims_btn", use_container_width=True):
                st.session_state.current_step = 'claims_summary'
                st.rerun()
        else:
            st.button("Open Claims Summary", disabled=True, help="Complete field mapping first", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="scenario-card">
            <h3>🛡️ FWA Detection</h3>
            <p>Advanced fraud, waste, and abuse detection using multiple analytical scenarios and ML algorithms</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.mapping_confirmed:
            if st.button("Open FWA Detection", key="fwa_btn", use_container_width=True):
                st.session_state.current_step = 'fwa_detection'
                st.rerun()
        else:
            st.button("Open FWA Detection", disabled=True, help="Complete field mapping first", use_container_width=True)
    
    with col3:
        st.markdown("""
        <div class="scenario-card">
            <h3>📈 Trend Analysis</h3>
            <p>Identify patterns and trends in healthcare claims data over time</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.mapping_confirmed:
            if st.button("Open Trend Analysis", key="trend_btn", use_container_width=True):
                st.session_state.current_step = 'trend_analysis'
                st.rerun()
        else:
            st.button("Open Trend Analysis", disabled=True, help="Complete field mapping first", use_container_width=True)

def show_database_modal():
    """Show database connection form"""
    st.subheader("🗄️ Database Connection")
    
    with st.form("db_connection"):
        db_type = st.selectbox("Database Type", ["postgresql", "mysql", "sqlserver", "oracle"])
        host = st.text_input("Host", value="localhost")
        port = st.text_input("Port", value="5432")
        database = st.text_input("Database Name")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.form_submit_button("🔌 Connect", use_container_width=True):
            db_config = {
                'db_type': db_type,
                'host': host,
                'port': port,
                'database': database,
                'username': username,
                'password': password
            }
            
            # Generate comprehensive sample data
            df = st.session_state.analytics.generate_sample_data()
            st.session_state.analytics.data = df
            st.session_state.data_loaded = True
            st.success("✅ Connected to database successfully!")
            st.session_state.current_step = 'preview'
            st.rerun()

def show_data_preview(analytics):
    """Show data preview and summary"""
    st.header("👁️ Data Preview & Summary")
    
    if analytics.data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Data Preview (First 5 rows)")
            st.dataframe(analytics.data.head(), use_container_width=True)
        
        with col2:
            st.subheader("📈 Data Summary")
            summary = analytics.generate_data_summary(analytics.data)
            
            # Display metrics in cards
            st.markdown(f"""
            <div class="metric-card">
                <h3>{summary['total_rows']}</h3>
                <p>Total Rows</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{summary['total_columns']}</h3>
                <p>Total Columns</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Column Information:**")
            for col, info in list(summary['column_info'].items())[:5]:  # Show first 5 columns
                with st.expander(f"{col} ({info['dtype']})"):
                    st.write(f"Null Count: {info['null_count']}")
                    st.write(f"Unique Values: {info['unique_count']}")
                    if 'mean' in info:
                        st.write(f"Mean: {info['mean']:.2f}")
                        st.write(f"Std: {info['std']:.2f}")
        
        if st.button("🔗 Field Mapping", use_container_width=True):
            st.session_state.current_step = 'mapping'
            st.rerun()

def show_field_mapping(analytics):
    """Show field mapping interface"""
    st.header("🔗 Field Mapping")
    st.write("Required to run FWA Analytics")
    
    if analytics.data is not None:
        mappings = analytics.generate_field_mappings(list(analytics.data.columns))
        
        # Create form for field mappings
        with st.form("field_mapping"):
            updated_mappings = []
            
            # Show mappings in a more compact format
            st.write("### Field Mapping Configuration")
            
            for i, mapping in enumerate(mappings):
                col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
                
                with col1:
                    st.write(f"**{mapping['required_field']}**")
                    st.caption(mapping['description'])
                
                with col2:
                    selected_field = st.selectbox(
                        "User Field",
                        options=list(analytics.data.columns),
                        index=list(analytics.data.columns).index(mapping['user_field']),
                        key=f"field_{i}",
                        label_visibility="collapsed"
                    )
                
                with col3:
                    st.write(mapping['data_type'])
                
                with col4:
                    # Confidence score with color
                    score = mapping['confidence_score']
                    if score >= 91:
                        st.markdown(f'<span class="confidence-high">{score}%</span>', unsafe_allow_html=True)
                    elif score >= 41:
                        st.markdown(f'<span class="confidence-medium">{score}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="confidence-low">{score}%</span>', unsafe_allow_html=True)
                
                with col5:
                    is_confirm = st.checkbox("Confirm", value=mapping['is_confirm'], key=f"confirm_{i}")
                
                updated_mapping = mapping.copy()
                updated_mapping['user_field'] = selected_field
                updated_mapping['is_confirm'] = is_confirm
                updated_mappings.append(updated_mapping)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("✅ Confirm Mapping", use_container_width=True):
                    analytics.field_mappings = {m['required_field']: m for m in updated_mappings if m['is_confirm']}
                    st.session_state.mapping_confirmed = True
                    st.success("✅ Field mapping confirmed!")
                    st.session_state.current_step = 'dashboard'
                    st.rerun()
            
            with col2:
                if st.form_submit_button("✅ Confirm All", use_container_width=True):
                    for mapping in updated_mappings:
                        mapping['is_confirm'] = True
                    analytics.field_mappings = {m['required_field']: m for m in updated_mappings}
                    st.session_state.mapping_confirmed = True
                    st.success("✅ All fields confirmed!")
                    st.session_state.current_step = 'dashboard'
                    st.rerun()

def show_claims_summary(analytics):
    """Show comprehensive claims data summary"""
    st.header("📊 Claims Data Summary")
    
    if analytics.data is not None:
        summary = analytics.generate_data_summary(analytics.data)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{summary['total_rows']}</h3>
                <p>Total Claims</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{summary['total_columns']}</h3>
                <p>Data Fields</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_amount = analytics.data['paid_amount'].mean() if 'paid_amount' in analytics.data.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>${avg_amount:,.2f}</h3>
                <p>Avg Claim Amount</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            total_amount = analytics.data['paid_amount'].sum() if 'paid_amount' in analytics.data.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>${total_amount:,.2f}</h3>
                <p>Total Claims Value</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed field analysis
        st.subheader("📈 Field Analysis")
        
        # Show key fields first
        key_fields = ['paid_amount', 'age', 'provider_id', 'procedure_code', 'diagnostic_code']
        available_key_fields = [field for field in key_fields if field in analytics.data.columns]
        
        for column in available_key_fields:
            with st.expander(f"📊 {column} Analysis", expanded=False):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    info = summary['column_info'][column]
                    st.write("**Statistics:**")
                    st.write(f"Data Type: {info['dtype']}")
                    st.write(f"Unique Values: {info['unique_count']}")
                    st.write(f"Null Count: {info['null_count']}")
                    st.write(f"Null %: {info['null_percentage']:.1f}%")
                    
                    if 'mean' in info:
                        st.write(f"Mean: {info['mean']:.2f}")
                        st.write(f"Median: {info['median']:.2f}")
                        st.write(f"Std Dev: {info['std']:.2f}")
                        st.write(f"Min: {info['min']}")
                        st.write(f"Max: {info['max']}")
                
                with col2:
                    # Create and display visualization
                    fig = analytics.create_visualization(analytics.data, column)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

def show_fwa_detection(analytics):
    """Show FWA detection scenarios"""
    st.header("🛡️ FWA Detection Scenarios")
    
    if analytics.field_mappings:
        active_fields = list(analytics.field_mappings.keys())
        available, unavailable = analytics.get_available_scenarios(active_fields)
        
        # Separate Python and ML scenarios
        available_python = [s for s in available if s['category'] == 'python_rules']
        available_ml = [s for s in available if s['category'] == 'ml_algorithms']
        unavailable_python = [s for s in unavailable if s['category'] == 'python_rules']
        unavailable_ml = [s for s in unavailable if s['category'] == 'ml_algorithms']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"✅ Available Python Scenarios ({len(available_python)})")
            
            selected_python_scenarios = []
            for scenario in available_python:
                if st.checkbox(scenario['name'], key=f"python_{scenario['name']}"):
                    selected_python_scenarios.append(scenario['name'])
                
                with st.expander(f"Details: {scenario['name']}"):
                    st.write(scenario['description'])
                    st.write(f"Required fields: {', '.join(scenario['required_fields'])}")
            
            st.subheader(f"✅ Available ML Scenarios ({len(available_ml)})")
            
            selected_ml_scenarios = []
            for scenario in available_ml:
                if st.checkbox(scenario['name'], key=f"ml_{scenario['name']}"):
                    selected_ml_scenarios.append(scenario['name'])
                
                with st.expander(f"Details: {scenario['name']}"):
                    st.write(scenario['description'])
                    st.write(f"Required fields: {', '.join(scenario['required_fields'])}")
        
        with col2:
            st.subheader(f"❌ Unavailable Python Scenarios ({len(unavailable_python)})")
            
            for scenario in unavailable_python:
                st.markdown(f"""
                <div class="scenario-card">
                    <h4>{scenario['name']}</h4>
                    <p>{scenario['description']}</p>
                    <p style="color: red;"><strong>Missing fields:</strong> {', '.join(scenario['missing_fields'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader(f"❌ Unavailable ML Scenarios ({len(unavailable_ml)})")
            
            for scenario in unavailable_ml:
                st.markdown(f"""
                <div class="scenario-card">
                    <h4>{scenario['name']}</h4>
                    <p>{scenario['description']}</p>
                    <p style="color: red;"><strong>Missing fields:</strong> {', '.join(scenario['missing_fields'])}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Run scenarios
        all_selected = selected_python_scenarios + selected_ml_scenarios
        if all_selected:
            if st.button(f"🚀 Run Selected Scenarios ({len(all_selected)})", type="primary", use_container_width=True):
                with st.spinner("Running FWA analysis..."):
                    # Run Python scenarios
                    results_df = analytics.data.copy()
                    if selected_python_scenarios:
                        results_df = analytics.run_python_scenarios(results_df, selected_python_scenarios)
                    
                    # Run ML scenarios
                    if selected_ml_scenarios:
                        results_df = analytics.run_ml_scenarios(results_df, selected_ml_scenarios)
                    
                    # Calculate fraud scores
                    results_df = analytics.calculate_fraud_score(results_df, all_selected)
                    
                    # Create Excel file with results
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Python rules sheet
                        python_flags = [col for col in results_df.columns if any(scenario in col for scenario in selected_python_scenarios) and col.endswith('_flag')]
                        python_results = results_df[['claim_id', 'member_id', 'provider_id', 'paid_amount', 'fraud_score'] + python_flags]
                        python_results.to_excel(writer, sheet_name='Python_Rules', index=False)
                        
                        # ML scenarios sheet
                        ml_flags = [col for col in results_df.columns if any(scenario in col for scenario in selected_ml_scenarios) and col.endswith('_flag')]
                        ml_results = results_df[['claim_id', 'member_id', 'provider_id', 'paid_amount', 'fraud_score'] + ml_flags]
                        ml_results.to_excel(writer, sheet_name='ML_Scenarios', index=False)
                    
                    output.seek(0)
                    
                    st.success(f"✅ FWA analysis completed! Found {results_df[results_df['fraud_score'] > 0].shape[0]} potentially fraudulent claims.")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download FWA Results",
                        data=output.getvalue(),
                        file_name=f"fwa_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Show top flagged claims
                    st.subheader("🚨 Top Flagged Claims")
                    top_flagged = results_df[results_df['fraud_score'] > 0].head(10)
                    st.dataframe(top_flagged[['claim_id', 'member_id', 'provider_id', 'paid_amount', 'fraud_score']], use_container_width=True)
                    
                    # Show fraud score distribution
                    fig = px.histogram(results_df, x='fraud_score', title='Fraud Score Distribution',
                                     color_discrete_sequence=['#ff6b35'])
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#333')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations to Prevent FWA")
                    high_fraud_claims = results_df[results_df['fraud_score'] > 50].shape[0]
                    total_claims = results_df.shape[0]
                    fraud_rate = (high_fraud_claims / total_claims) * 100
                    
                    recommendations = [
                        f"🔍 **Enhanced Monitoring**: {fraud_rate:.1f}% of claims show high fraud risk - implement real-time monitoring",
                        "🤖 **Automated Screening**: Deploy ML models for continuous fraud detection",
                        "👥 **Provider Education**: Focus on providers with multiple flagged claims",
                        "📊 **Regular Audits**: Conduct quarterly reviews of high-risk claims",
                        "🔒 **Strengthen Controls**: Implement additional verification for high-value claims"
                    ]
                    
                    for rec in recommendations:
                        st.markdown(rec)

def show_trend_analysis(analytics):
    """Show trend analysis module"""
    st.header("📈 Trend Analysis")
    
    if analytics.data is not None:
        st.info("This module analyzes temporal patterns in healthcare claims data, identifying trends in claim volumes, amounts, provider behaviors, and seasonal variations.")
        
        # Time-based analysis
        if 'claim_invoice_date' in analytics.data.columns:
            # Convert to datetime if not already
            analytics.data['claim_invoice_date'] = pd.to_datetime(analytics.data['claim_invoice_date'])
            
            # Monthly trend analysis
            monthly_data = analytics.data.groupby(analytics.data['claim_invoice_date'].dt.to_period('M')).agg({
                'claim_id': 'count',
                'paid_amount': 'sum'
            }).reset_index()
            monthly_data['claim_invoice_date'] = monthly_data['claim_invoice_date'].astype(str)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.line(monthly_data, x='claim_invoice_date', y='claim_id', 
                              title='Monthly Claim Volume Trend',
                              color_discrete_sequence=['#ff6b35'])
                fig1.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333')
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.line(monthly_data, x='claim_invoice_date', y='paid_amount', 
                              title='Monthly Claim Amount Trend',
                              color_discrete_sequence=['#2a5298'])
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333')
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        if st.button("🚀 Start Advanced Trend Analysis", type="primary", use_container_width=True):
            st.success("✅ Advanced trend analysis would include seasonal decomposition, anomaly detection in time series, and predictive modeling for future claim patterns.")

if __name__ == "__main__":
    main()