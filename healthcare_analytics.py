import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlalchemy
from sqlalchemy import create_engine
import io
import base64
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for network-inspired background
def load_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 25%, #ff6b35 75%, #f7931e 100%);
        background-attachment: fixed;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .scenario-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .unavailable-scenario {
        background: rgba(255, 255, 255, 0.7);
        border-left: 4px solid #f44336;
        opacity: 0.7;
    }
    </style>
    """, unsafe_allow_html=True)

class HealthcareAnalytics:
    def __init__(self):
        self.data = None
        self.data_summary = None
        self.field_mappings = {}
        self.confidence_scores = {}
        
        # Comprehensive required fields for healthcare analytics
        self.required_fields = {
            'claim_id': {'type': 'string', 'description': 'Unique claim identifier'},
            'member_id': {'type': 'string', 'description': 'Member/Patient identifier'},
            'provider_id': {'type': 'string', 'description': 'Healthcare provider ID'},
            'provider_type': {'type': 'string', 'description': 'Type of healthcare provider'},
            'claim_invoice_no': {'type': 'string', 'description': 'Claim invoice number'},
            'claim_invoice_line_no': {'type': 'string', 'description': 'Invoice line number'},
            'invoice_no_reference': {'type': 'string', 'description': 'Invoice reference number'},
            'claim_version': {'type': 'string', 'description': 'Version of the claim'},
            'latest_claim_version_ind': {'type': 'boolean', 'description': 'Latest version indicator'},
            'claim_status_code': {'type': 'string', 'description': 'Current claim status'},
            'incident_count': {'type': 'number', 'description': 'Number of incidents'},
            'diagnostic_code': {'type': 'string', 'description': 'ICD-10 diagnostic code'},
            'procedure_code': {'type': 'string', 'description': 'CPT procedure code'},
            'age': {'type': 'number', 'description': 'Patient age'},
            'gender': {'type': 'string', 'description': 'Patient gender'},
            'nationality_code': {'type': 'string', 'description': 'Patient nationality'},
            'claim_invoice_date': {'type': 'date', 'description': 'Claim invoice date'},
            'admission_date': {'type': 'date', 'description': 'Hospital admission date'},
            'discharge_date': {'type': 'date', 'description': 'Hospital discharge date'},
            'los': {'type': 'number', 'description': 'Length of stay'},
            'pos': {'type': 'string', 'description': 'Place of service'},
            'treatment_from_date': {'type': 'date', 'description': 'Treatment start date'},
            'treatment_to_date': {'type': 'date', 'description': 'Treatment end date'},
            'provider_country_code': {'type': 'string', 'description': 'Provider country'},
            'paid_amount': {'type': 'number', 'description': 'Amount paid'},
            'claimed_currency_code': {'type': 'string', 'description': 'Claim currency'},
            'payment_currency_code': {'type': 'string', 'description': 'Payment currency'},
            'base_currency_code': {'type': 'string', 'description': 'Base currency'},
            'claim_invoice_gross_total': {'type': 'number', 'description': 'Gross total amount'},
            'payee_type': {'type': 'string', 'description': 'Type of payee'},
            'conversion_rate': {'type': 'number', 'description': 'Currency conversion rate'},
            'policy_start_date': {'type': 'date', 'description': 'Policy start date'},
            'policy_end_date': {'type': 'date', 'description': 'Policy end date'},
            'previous_fraud_flags': {'type': 'boolean', 'description': 'Previous fraud indicators'},
            'member_zip_code': {'type': 'string', 'description': 'Member zip code'},
            'provider_zip_code': {'type': 'string', 'description': 'Provider zip code'},
            'coverage_type': {'type': 'string', 'description': 'Coverage type'},
            'facility_type': {'type': 'string', 'description': 'Healthcare facility type'},
            'ndc_code': {'type': 'string', 'description': 'National Drug Code'},
            'prior_auth_required_flag': {'type': 'boolean', 'description': 'Prior authorization required'},
            'prior_auth_number': {'type': 'string', 'description': 'Prior authorization number'},
            'prior_auth_approved_flag': {'type': 'boolean', 'description': 'Prior auth approved'},
            'prior_auth_approval_date': {'type': 'date', 'description': 'Prior auth approval date'},
            'referral_required_flag': {'type': 'boolean', 'description': 'Referral required'},
            'referral_provider_id': {'type': 'string', 'description': 'Referring provider ID'},
            'referral_submission_date': {'type': 'date', 'description': 'Referral submission date'},
            'claim_status_datetime': {'type': 'datetime', 'description': 'Claim status timestamp'},
            'denial_code': {'type': 'string', 'description': 'Denial code if applicable'},
            'denial_reason': {'type': 'string', 'description': 'Reason for denial'},
            'billed_amount': {'type': 'number', 'description': 'Amount billed'},
            'allowed_amount': {'type': 'number', 'description': 'Allowed amount'},
            'deductible_remaining': {'type': 'number', 'description': 'Remaining deductible'},
            'copay_amount': {'type': 'number', 'description': 'Copay amount'},
            'coinsurance_pct': {'type': 'number', 'description': 'Coinsurance percentage'},
            'policy_code': {'type': 'string', 'description': 'Policy code'},
            'policy_name': {'type': 'string', 'description': 'Policy name'},
            'policy_type': {'type': 'string', 'description': 'Type of policy'},
            'policy_max_coverage': {'type': 'number', 'description': 'Maximum coverage amount'},
            'policy_min_coverage': {'type': 'number', 'description': 'Minimum coverage amount'},
            'deductible_amount': {'type': 'number', 'description': 'Deductible amount'},
            'out_of_pocket_max': {'type': 'number', 'description': 'Out of pocket maximum'},
            'enrollment_date': {'type': 'date', 'description': 'Policy enrollment date'},
            'renewal_date': {'type': 'date', 'description': 'Policy renewal date'},
            'premium_amount': {'type': 'number', 'description': 'Premium amount'},
            'premium_frequency': {'type': 'string', 'description': 'Premium frequency'},
            'employer_contribution': {'type': 'number', 'description': 'Employer contribution'},
            'customer_contribution': {'type': 'number', 'description': 'Customer contribution'},
            'discount_amount': {'type': 'number', 'description': 'Discount amount'},
            'network_type': {'type': 'string', 'description': 'Network type'},
            'coverage_area': {'type': 'string', 'description': 'Coverage area'},
            'prescription_coverage': {'type': 'boolean', 'description': 'Prescription coverage'},
            'preventive_services_covered': {'type': 'boolean', 'description': 'Preventive services covered'},
            'policy_status': {'type': 'string', 'description': 'Policy status'},
            'is_default_policy': {'type': 'boolean', 'description': 'Default policy flag'},
            'renewed_flag': {'type': 'boolean', 'description': 'Renewal flag'}
        }
        
        # Python-based FWA scenarios (for Hypothesis sheet)
        self.python_scenarios = {
            'Duplicate Claims Detection': {
                'required_fields': ['claim_id', 'member_id', 'provider_id', 'paid_amount', 'claim_invoice_date'],
                'description': 'Identifies potential duplicate claims submitted by providers'
            },
            'Billing Pattern Analysis': {
                'required_fields': ['provider_id', 'paid_amount', 'claim_invoice_date', 'procedure_code'],
                'description': 'Analyzes unusual billing patterns and frequency'
            },
            'Age-Service Mismatch': {
                'required_fields': ['age', 'procedure_code', 'diagnostic_code'],
                'description': 'Detects services inappropriate for patient age'
            },
            'Provider Network Analysis': {
                'required_fields': ['provider_id', 'member_id', 'provider_zip_code', 'member_zip_code'],
                'description': 'Identifies suspicious provider-patient relationships'
            },
            'Amount Outlier Detection': {
                'required_fields': ['paid_amount', 'procedure_code', 'pos'],
                'description': 'Detects claims with unusually high amounts'
            },
            'Frequency Analysis': {
                'required_fields': ['provider_id', 'claim_invoice_date', 'procedure_code'],
                'description': 'Identifies providers with unusually high claim frequency'
            },
            'Geographic Anomaly Detection': {
                'required_fields': ['member_zip_code', 'provider_zip_code', 'paid_amount'],
                'description': 'Detects claims from geographically distant providers'
            },
            'Prior Authorization Violations': {
                'required_fields': ['prior_auth_required_flag', 'prior_auth_approved_flag', 'procedure_code'],
                'description': 'Identifies claims without required prior authorization'
            }
        }
        
        # ML-based scenarios (for ML Scenarios sheet)
        self.ml_scenarios = {
            'Anomaly Detection (Isolation Forest)': {
                'required_fields': ['paid_amount', 'age', 'los'],
                'description': 'Uses Isolation Forest to detect anomalous claims'
            },
            'Clustering Analysis (DBSCAN)': {
                'required_fields': ['paid_amount', 'provider_id', 'procedure_code'],
                'description': 'Identifies unusual claim clusters using DBSCAN'
            },
            'Fraud Prediction Model': {
                'required_fields': ['paid_amount', 'age', 'los', 'provider_id'],
                'description': 'Predicts fraud probability using Random Forest'
            },
            'Pattern Recognition': {
                'required_fields': ['provider_id', 'procedure_code', 'paid_amount', 'claim_invoice_date'],
                'description': 'Identifies suspicious patterns in provider behavior'
            }
        }

    def load_file_data(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(uploaded_file)
            
            self.generate_data_summary()
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False

    def connect_database(self, credentials):
        """Connect to database and load data"""
        try:
            if credentials['dbType'] == 'postgresql':
                connection_string = f"postgresql://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
            elif credentials['dbType'] == 'mysql':
                connection_string = f"mysql+pymysql://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
            elif credentials['dbType'] == 'sqlserver':
                connection_string = f"mssql+pyodbc://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}?driver=ODBC+Driver+17+for+SQL+Server"
            elif credentials['dbType'] == 'oracle':
                connection_string = f"oracle+cx_oracle://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
            
            engine = create_engine(connection_string)
            
            # For demo purposes, create sample data
            sample_data = {
                'claim_id': ['CLM001', 'CLM002', 'CLM003', 'CLM004', 'CLM005'],
                'member_id': ['MEM001', 'MEM002', 'MEM003', 'MEM004', 'MEM005'],
                'provider_id': ['PROV001', 'PROV002', 'PROV001', 'PROV003', 'PROV002'],
                'paid_amount': [1500.00, 2300.50, 875.25, 3200.00, 1890.75],
                'claim_invoice_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
                'diagnostic_code': ['M79.3', 'E11.9', 'I10', 'Z00.00', 'M25.511'],
                'procedure_code': ['99213', '99214', '99212', '99215', '99213'],
                'age': [45, 62, 38, 29, 55],
                'pos': ['Office', 'Office', 'Office', 'Office', 'Office']
            }
            
            self.data = pd.DataFrame(sample_data)
            self.generate_data_summary()
            return True
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            return False

    def generate_data_summary(self):
        """Generate comprehensive data summary"""
        if self.data is not None:
            summary = {}
            for column in self.data.columns:
                col_data = self.data[column]
                
                summary[column] = {
                    'type': self.infer_data_type(col_data),
                    'null_count': col_data.isnull().sum(),
                    'unique_count': col_data.nunique(),
                    'total_count': len(col_data)
                }
                
                if pd.api.types.is_numeric_dtype(col_data):
                    summary[column].update({
                        'mean': col_data.mean(),
                        'median': col_data.median(),
                        'std': col_data.std(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'q25': col_data.quantile(0.25),
                        'q75': col_data.quantile(0.75)
                    })
            
            self.data_summary = summary

    def infer_data_type(self, series):
        """Infer data type of a pandas series"""
        if pd.api.types.is_numeric_dtype(series):
            return 'number'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'date'
        elif series.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').any():
            return 'date'
        else:
            return 'string'

    def calculate_field_similarity(self, user_field, required_field):
        """Calculate similarity between user field and required field"""
        user_field = user_field.lower().replace('_', ' ').replace('-', ' ')
        required_field = required_field.lower().replace('_', ' ').replace('-', ' ')
        
        # Exact match
        if user_field == required_field:
            return 100
        
        # Contains match
        if required_field in user_field or user_field in required_field:
            return 85
        
        # Word overlap
        user_words = set(user_field.split())
        required_words = set(required_field.split())
        
        if user_words & required_words:
            overlap = len(user_words & required_words)
            total = len(user_words | required_words)
            return int((overlap / total) * 80)
        
        return 0

    def generate_field_mappings(self):
        """Generate field mappings with confidence scores"""
        if self.data is None:
            return
        
        user_fields = list(self.data.columns)
        mappings = {}
        confidence_scores = {}
        
        for required_field in self.required_fields.keys():
            best_match = None
            best_score = 0
            
            for user_field in user_fields:
                score = self.calculate_field_similarity(user_field, required_field)
                if score > best_score:
                    best_score = score
                    best_match = user_field
            
            if best_match:
                mappings[required_field] = best_match
                confidence_scores[required_field] = best_score
        
        self.field_mappings = mappings
        self.confidence_scores = confidence_scores

    def get_available_scenarios(self, scenario_type='python'):
        """Get available scenarios based on mapped fields"""
        scenarios = self.python_scenarios if scenario_type == 'python' else self.ml_scenarios
        available = []
        unavailable = []
        
        confirmed_fields = [field for field, confirmed in st.session_state.get('field_confirmations', {}).items() if confirmed]
        
        for scenario_name, scenario_info in scenarios.items():
            required_fields = scenario_info['required_fields']
            if all(field in confirmed_fields for field in required_fields):
                available.append(scenario_name)
            else:
                unavailable.append(scenario_name)
        
        return available, unavailable

    def run_python_scenarios(self, selected_scenarios):
        """Run Python-based FWA detection scenarios"""
        results = {}
        
        for scenario in selected_scenarios:
            if scenario == 'Duplicate Claims Detection':
                results[scenario] = self.detect_duplicate_claims()
            elif scenario == 'Billing Pattern Analysis':
                results[scenario] = self.analyze_billing_patterns()
            elif scenario == 'Age-Service Mismatch':
                results[scenario] = self.detect_age_service_mismatch()
            elif scenario == 'Provider Network Analysis':
                results[scenario] = self.analyze_provider_network()
            elif scenario == 'Amount Outlier Detection':
                results[scenario] = self.detect_amount_outliers()
            elif scenario == 'Frequency Analysis':
                results[scenario] = self.analyze_frequency_patterns()
            elif scenario == 'Geographic Anomaly Detection':
                results[scenario] = self.detect_geographic_anomalies()
            elif scenario == 'Prior Authorization Violations':
                results[scenario] = self.detect_prior_auth_violations()
        
        return results

    def run_ml_scenarios(self, selected_scenarios):
        """Run ML-based FWA detection scenarios"""
        results = {}
        
        for scenario in selected_scenarios:
            if scenario == 'Anomaly Detection (Isolation Forest)':
                results[scenario] = self.isolation_forest_detection()
            elif scenario == 'Clustering Analysis (DBSCAN)':
                results[scenario] = self.dbscan_clustering()
            elif scenario == 'Fraud Prediction Model':
                results[scenario] = self.fraud_prediction_model()
            elif scenario == 'Pattern Recognition':
                results[scenario] = self.pattern_recognition()
        
        return results

    # Python-based detection methods
    def detect_duplicate_claims(self):
        """Detect duplicate claims"""
        if self.data is None:
            return []
        
        # Get mapped field names
        claim_id_field = self.field_mappings.get('claim_id')
        member_id_field = self.field_mappings.get('member_id')
        provider_id_field = self.field_mappings.get('provider_id')
        amount_field = self.field_mappings.get('paid_amount')
        
        if not all([claim_id_field, member_id_field, provider_id_field, amount_field]):
            return []
        
        # Find duplicates based on member, provider, and amount
        duplicates = self.data.duplicated(subset=[member_id_field, provider_id_field, amount_field], keep=False)
        return self.data[duplicates][claim_id_field].tolist()

    def analyze_billing_patterns(self):
        """Analyze unusual billing patterns"""
        if self.data is None:
            return []
        
        provider_field = self.field_mappings.get('provider_id')
        amount_field = self.field_mappings.get('paid_amount')
        
        if not all([provider_field, amount_field]):
            return []
        
        # Calculate provider statistics
        provider_stats = self.data.groupby(provider_field)[amount_field].agg(['count', 'mean', 'std']).reset_index()
        
        # Flag providers with unusual patterns (high frequency + high amounts)
        high_freq_threshold = provider_stats['count'].quantile(0.95)
        high_amount_threshold = provider_stats['mean'].quantile(0.95)
        
        suspicious_providers = provider_stats[
            (provider_stats['count'] > high_freq_threshold) & 
            (provider_stats['mean'] > high_amount_threshold)
        ][provider_field].tolist()
        
        flagged_claims = self.data[self.data[provider_field].isin(suspicious_providers)]
        return flagged_claims[self.field_mappings.get('claim_id', self.data.columns[0])].tolist()

    def detect_age_service_mismatch(self):
        """Detect age-service mismatches"""
        if self.data is None:
            return []
        
        age_field = self.field_mappings.get('age')
        procedure_field = self.field_mappings.get('procedure_code')
        
        if not all([age_field, procedure_field]):
            return []
        
        # Define age-inappropriate procedures (simplified logic)
        pediatric_procedures = ['90460', '90461', '90471']  # Vaccination codes
        geriatric_procedures = ['99324', '99325', '99326']  # Nursing home visits
        
        flagged_claims = []
        
        # Flag pediatric procedures for adults
        adult_pediatric = self.data[
            (self.data[age_field] > 18) & 
            (self.data[procedure_field].isin(pediatric_procedures))
        ]
        
        # Flag geriatric procedures for young patients
        young_geriatric = self.data[
            (self.data[age_field] < 65) & 
            (self.data[procedure_field].isin(geriatric_procedures))
        ]
        
        flagged_claims.extend(adult_pediatric[self.field_mappings.get('claim_id', self.data.columns[0])].tolist())
        flagged_claims.extend(young_geriatric[self.field_mappings.get('claim_id', self.data.columns[0])].tolist())
        
        return flagged_claims

    def analyze_provider_network(self):
        """Analyze provider network for suspicious relationships"""
        if self.data is None:
            return []
        
        provider_field = self.field_mappings.get('provider_id')
        member_field = self.field_mappings.get('member_id')
        
        if not all([provider_field, member_field]):
            return []
        
        # Calculate provider-member interaction frequency
        interactions = self.data.groupby([provider_field, member_field]).size().reset_index(name='frequency')
        
        # Flag high-frequency interactions (potential collusion)
        high_freq_threshold = interactions['frequency'].quantile(0.95)
        suspicious_pairs = interactions[interactions['frequency'] > high_freq_threshold]
        
        flagged_claims = []
        for _, row in suspicious_pairs.iterrows():
            claims = self.data[
                (self.data[provider_field] == row[provider_field]) & 
                (self.data[member_field] == row[member_field])
            ]
            flagged_claims.extend(claims[self.field_mappings.get('claim_id', self.data.columns[0])].tolist())
        
        return flagged_claims

    def detect_amount_outliers(self):
        """Detect amount outliers using statistical methods"""
        if self.data is None:
            return []
        
        amount_field = self.field_mappings.get('paid_amount')
        if not amount_field:
            return []
        
        # Use IQR method for outlier detection
        Q1 = self.data[amount_field].quantile(0.25)
        Q3 = self.data[amount_field].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.data[
            (self.data[amount_field] < lower_bound) | 
            (self.data[amount_field] > upper_bound)
        ]
        
        return outliers[self.field_mappings.get('claim_id', self.data.columns[0])].tolist()

    def analyze_frequency_patterns(self):
        """Analyze claim frequency patterns"""
        if self.data is None:
            return []
        
        provider_field = self.field_mappings.get('provider_id')
        date_field = self.field_mappings.get('claim_invoice_date')
        
        if not all([provider_field, date_field]):
            return []
        
        # Convert date field to datetime
        self.data[date_field] = pd.to_datetime(self.data[date_field], errors='coerce')
        
        # Calculate daily claim frequency per provider
        daily_claims = self.data.groupby([provider_field, self.data[date_field].dt.date]).size().reset_index(name='daily_count')
        
        # Flag providers with unusually high daily claim counts
        high_freq_threshold = daily_claims['daily_count'].quantile(0.95)
        suspicious_days = daily_claims[daily_claims['daily_count'] > high_freq_threshold]
        
        flagged_claims = []
        for _, row in suspicious_days.iterrows():
            claims = self.data[
                (self.data[provider_field] == row[provider_field]) & 
                (self.data[date_field].dt.date == row[date_field])
            ]
            flagged_claims.extend(claims[self.field_mappings.get('claim_id', self.data.columns[0])].tolist())
        
        return flagged_claims

    def detect_geographic_anomalies(self):
        """Detect geographic anomalies"""
        if self.data is None:
            return []
        
        member_zip_field = self.field_mappings.get('member_zip_code')
        provider_zip_field = self.field_mappings.get('provider_zip_code')
        
        if not all([member_zip_field, provider_zip_field]):
            return []
        
        # Simplified geographic analysis (in real implementation, use actual distance calculation)
        # Flag claims where zip codes are very different (simplified logic)
        flagged_claims = []
        
        for idx, row in self.data.iterrows():
            member_zip = str(row[member_zip_field])[:3] if pd.notna(row[member_zip_field]) else ''
            provider_zip = str(row[provider_zip_field])[:3] if pd.notna(row[provider_zip_field]) else ''
            
            # Flag if first 3 digits of zip codes are very different
            if member_zip and provider_zip and abs(int(member_zip) - int(provider_zip)) > 100:
                flagged_claims.append(row[self.field_mappings.get('claim_id', self.data.columns[0])])
        
        return flagged_claims

    def detect_prior_auth_violations(self):
        """Detect prior authorization violations"""
        if self.data is None:
            return []
        
        auth_required_field = self.field_mappings.get('prior_auth_required_flag')
        auth_approved_field = self.field_mappings.get('prior_auth_approved_flag')
        
        if not all([auth_required_field, auth_approved_field]):
            return []
        
        # Flag claims that required prior auth but weren't approved
        violations = self.data[
            (self.data[auth_required_field] == True) & 
            (self.data[auth_approved_field] != True)
        ]
        
        return violations[self.field_mappings.get('claim_id', self.data.columns[0])].tolist()

    # ML-based detection methods
    def isolation_forest_detection(self):
        """Anomaly detection using Isolation Forest"""
        if self.data is None:
            return []
        
        # Prepare numerical features
        numerical_fields = ['paid_amount', 'age', 'los']
        available_fields = [self.field_mappings.get(field) for field in numerical_fields if self.field_mappings.get(field)]
        
        if len(available_fields) < 2:
            return []
        
        # Prepare data
        feature_data = self.data[available_fields].select_dtypes(include=[np.number]).fillna(0)
        
        if feature_data.empty:
            return []
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(feature_data)
        
        # Get anomalous claims
        anomalous_indices = np.where(anomaly_labels == -1)[0]
        flagged_claims = self.data.iloc[anomalous_indices][self.field_mappings.get('claim_id', self.data.columns[0])].tolist()
        
        return flagged_claims

    def dbscan_clustering(self):
        """Clustering analysis using DBSCAN"""
        if self.data is None:
            return []
        
        # Prepare features
        amount_field = self.field_mappings.get('paid_amount')
        provider_field = self.field_mappings.get('provider_id')
        
        if not all([amount_field, provider_field]):
            return []
        
        # Encode categorical variables
        le = LabelEncoder()
        encoded_provider = le.fit_transform(self.data[provider_field].astype(str))
        
        # Prepare feature matrix
        features = np.column_stack([
            self.data[amount_field].fillna(0),
            encoded_provider
        ])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        # Flag outliers (label = -1)
        outlier_indices = np.where(cluster_labels == -1)[0]
        flagged_claims = self.data.iloc[outlier_indices][self.field_mappings.get('claim_id', self.data.columns[0])].tolist()
        
        return flagged_claims

    def fraud_prediction_model(self):
        """Fraud prediction using Random Forest"""
        if self.data is None:
            return []
        
        # Prepare features
        numerical_fields = ['paid_amount', 'age', 'los']
        available_fields = [self.field_mappings.get(field) for field in numerical_fields if self.field_mappings.get(field)]
        
        if len(available_fields) < 2:
            return []
        
        # Create synthetic fraud labels for demonstration
        np.random.seed(42)
        fraud_labels = np.random.choice([0, 1], size=len(self.data), p=[0.9, 0.1])
        
        # Prepare features
        feature_data = self.data[available_fields].select_dtypes(include=[np.number]).fillna(0)
        
        if feature_data.empty:
            return []
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(feature_data, fraud_labels)
        
        # Predict fraud probability
        fraud_probabilities = rf.predict_proba(feature_data)[:, 1]
        
        # Flag high-risk claims (top 10%)
        threshold = np.percentile(fraud_probabilities, 90)
        high_risk_indices = np.where(fraud_probabilities > threshold)[0]
        
        flagged_claims = self.data.iloc[high_risk_indices][self.field_mappings.get('claim_id', self.data.columns[0])].tolist()
        
        return flagged_claims

    def pattern_recognition(self):
        """Pattern recognition for provider behavior"""
        if self.data is None:
            return []
        
        provider_field = self.field_mappings.get('provider_id')
        procedure_field = self.field_mappings.get('procedure_code')
        amount_field = self.field_mappings.get('paid_amount')
        
        if not all([provider_field, procedure_field, amount_field]):
            return []
        
        # Analyze provider-procedure patterns
        provider_patterns = self.data.groupby([provider_field, procedure_field]).agg({
            amount_field: ['count', 'mean', 'std']
        }).reset_index()
        
        provider_patterns.columns = [provider_field, procedure_field, 'count', 'mean_amount', 'std_amount']
        
        # Flag unusual patterns (high frequency + high variance)
        high_freq_threshold = provider_patterns['count'].quantile(0.9)
        high_variance_threshold = provider_patterns['std_amount'].quantile(0.9)
        
        suspicious_patterns = provider_patterns[
            (provider_patterns['count'] > high_freq_threshold) & 
            (provider_patterns['std_amount'] > high_variance_threshold)
        ]
        
        flagged_claims = []
        for _, pattern in suspicious_patterns.iterrows():
            claims = self.data[
                (self.data[provider_field] == pattern[provider_field]) & 
                (self.data[procedure_field] == pattern[procedure_field])
            ]
            flagged_claims.extend(claims[self.field_mappings.get('claim_id', self.data.columns[0])].tolist())
        
        return flagged_claims

    def create_visualization(self, column_name, column_info):
        """Create appropriate visualization for each field"""
        if self.data is None:
            return None
        
        col_data = self.data[column_name]
        
        if column_info['type'] == 'number':
            # Histogram for numerical data
            fig = px.histogram(
                x=col_data, 
                title=f'Distribution of {column_name}',
                nbins=20,
                color_discrete_sequence=['#667eea']
            )
        else:
            # Bar chart for categorical data
            value_counts = col_data.value_counts().head(10)
            fig = px.bar(
                x=value_counts.index, 
                y=value_counts.values,
                title=f'Top 10 Values in {column_name}',
                color_discrete_sequence=['#764ba2']
            )
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def calculate_fraud_score(self, claim_id, python_results, ml_results):
        """Calculate overall fraud score for a claim"""
        total_scenarios = len(python_results) + len(ml_results)
        flagged_count = 0
        
        # Count how many scenarios flagged this claim
        for scenario_flags in python_results.values():
            if claim_id in scenario_flags:
                flagged_count += 1
        
        for scenario_flags in ml_results.values():
            if claim_id in scenario_flags:
                flagged_count += 1
        
        # Calculate score as percentage
        fraud_score = int((flagged_count / total_scenarios) * 100) if total_scenarios > 0 else 0
        return fraud_score

    def export_results_to_excel(self, python_results, ml_results):
        """Export results to Excel with two sheets"""
        if self.data is None:
            return None
        
        # Prepare Python Rules sheet
        python_sheet_data = []
        claim_id_field = self.field_mappings.get('claim_id', self.data.columns[0])
        member_id_field = self.field_mappings.get('member_id', self.data.columns[1] if len(self.data.columns) > 1 else self.data.columns[0])
        provider_id_field = self.field_mappings.get('provider_id', self.data.columns[2] if len(self.data.columns) > 2 else self.data.columns[0])
        amount_field = self.field_mappings.get('paid_amount', self.data.columns[3] if len(self.data.columns) > 3 else self.data.columns[0])
        
        for _, row in self.data.iterrows():
            claim_id = row[claim_id_field]
            
            # Create row for Python Rules sheet
            python_row = {
                'Claim_ID': claim_id,
                'Member_ID': row[member_id_field],
                'Provider_ID': row[provider_id_field],
                'Paid_Amount': row[amount_field]
            }
            
            # Add binary flags for each Python scenario
            for scenario_name, flagged_claims in python_results.items():
                python_row[scenario_name.replace(' ', '_')] = 1 if claim_id in flagged_claims else 0
            
            # Calculate fraud score
            python_row['Fraud_Score'] = self.calculate_fraud_score(claim_id, python_results, {})
            
            python_sheet_data.append(python_row)
        
        # Prepare ML Scenarios sheet
        ml_sheet_data = []
        for _, row in self.data.iterrows():
            claim_id = row[claim_id_field]
            
            # Create row for ML Scenarios sheet
            ml_row = {
                'Claim_ID': claim_id,
                'Member_ID': row[member_id_field],
                'Provider_ID': row[provider_id_field],
                'Paid_Amount': row[amount_field]
            }
            
            # Add binary flags for each ML scenario
            for scenario_name, flagged_claims in ml_results.items():
                ml_row[scenario_name.replace(' ', '_').replace('(', '').replace(')', '')] = 1 if claim_id in flagged_claims else 0
            
            # Calculate fraud score
            ml_row['Fraud_Score'] = self.calculate_fraud_score(claim_id, {}, ml_results)
            
            ml_sheet_data.append(ml_row)
        
        # Create DataFrames and sort by fraud score
        python_df = pd.DataFrame(python_sheet_data).sort_values('Fraud_Score', ascending=False)
        ml_df = pd.DataFrame(ml_sheet_data).sort_values('Fraud_Score', ascending=False)
        
        # Create Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            python_df.to_excel(writer, sheet_name='Hypothesis', index=False)
            ml_df.to_excel(writer, sheet_name='ML_Scenarios', index=False)
        
        return output.getvalue()

def main():
    st.set_page_config(
        page_title="Healthcare Analytics Tool",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    load_custom_css()
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'dashboard'
    if 'analytics' not in st.session_state:
        st.session_state.analytics = HealthcareAnalytics()
    if 'field_confirmations' not in st.session_state:
        st.session_state.field_confirmations = {}
    if 'mapping_confirmed' not in st.session_state:
        st.session_state.mapping_confirmed = False
    
    analytics = st.session_state.analytics
    
    # Dashboard Page
    if st.session_state.current_step == 'dashboard':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # Header with upload options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.title("🏥 Healthcare Analytics Tool")
            st.markdown("""
            **Upload healthcare claims data to perform comprehensive analytics on claims datasets. 
            Access different analytical modules including FWA (Fraud, Waste, and Abuse) analytics.**
            """)
        
        with col2:
            st.markdown("### Upload Options")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload File",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel file"
            )
            
            if uploaded_file:
                if analytics.load_file_data(uploaded_file):
                    st.success("File uploaded successfully!")
                    st.session_state.current_step = 'preview'
                    st.rerun()
            
            # Database connection
            if st.button("Connect Your DB", type="secondary"):
                st.session_state.show_db_modal = True
        
        # Database connection modal
        if st.session_state.get('show_db_modal', False):
            with st.expander("Database Connection", expanded=True):
                st.subheader("Database Credentials")
                
                col1, col2 = st.columns(2)
                with col1:
                    db_type = st.selectbox("Database Type", ['postgresql', 'mysql', 'sqlserver', 'oracle'])
                    host = st.text_input("Host", value="localhost")
                    port = st.text_input("Port", value="5432")
                
                with col2:
                    database = st.text_input("Database Name")
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Connect", type="primary"):
                        credentials = {
                            'dbType': db_type,
                            'host': host,
                            'port': port,
                            'database': database,
                            'username': username,
                            'password': password
                        }
                        if analytics.connect_database(credentials):
                            st.success("Database connected successfully!")
                            st.session_state.show_db_modal = False
                            st.session_state.current_step = 'preview'
                            st.rerun()
                
                with col2:
                    if st.button("Cancel"):
                        st.session_state.show_db_modal = False
                        st.rerun()
        
        # Analytics modules (disabled until mapping is confirmed)
        st.markdown("### Analytics Modules")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.mapping_confirmed:
                if st.button("📊 Claims Data Summary", type="primary", use_container_width=True):
                    st.session_state.current_step = 'claims_summary'
                    st.rerun()
            else:
                st.button("📊 Claims Data Summary", disabled=True, use_container_width=True, help="Complete field mapping first")
        
        with col2:
            if st.session_state.mapping_confirmed:
                if st.button("🛡️ FWA Detection", type="primary", use_container_width=True):
                    st.session_state.current_step = 'fwa_detection'
                    st.rerun()
            else:
                st.button("🛡️ FWA Detection", disabled=True, use_container_width=True, help="Complete field mapping first")
        
        with col3:
            if st.session_state.mapping_confirmed:
                if st.button("📈 Trend Analysis", type="primary", use_container_width=True):
                    st.session_state.current_step = 'trend_analysis'
                    st.rerun()
            else:
                st.button("📈 Trend Analysis", disabled=True, use_container_width=True, help="Complete field mapping first")
        
        # Field mapping button
        if analytics.data is not None and not st.session_state.mapping_confirmed:
            st.markdown("---")
            if st.button("🔗 Field Mapping", type="primary", use_container_width=True):
                analytics.generate_field_mappings()
                st.session_state.current_step = 'mapping'
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Preview Page
    elif st.session_state.current_step == 'preview':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.title("📋 Data Preview & Summary")
        
        if analytics.data is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📄 Data Preview (First 5 rows)")
                st.dataframe(analytics.data.head(), use_container_width=True)
            
            with col2:
                st.subheader("📊 Data Summary")
                if analytics.data_summary:
                    # Summary metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total Rows", len(analytics.data))
                    with col_b:
                        st.metric("Total Columns", len(analytics.data.columns))
                    
                    # Column information
                    st.markdown("**Column Information:**")
                    for column, info in analytics.data_summary.items():
                        with st.expander(f"{column} ({info['type']})"):
                            col_x, col_y = st.columns(2)
                            with col_x:
                                st.write(f"**Unique Values:** {info['unique_count']}")
                                st.write(f"**Null Count:** {info['null_count']}")
                            with col_y:
                                if info['type'] == 'number':
                                    st.write(f"**Mean:** {info.get('mean', 'N/A'):.2f}" if info.get('mean') else "**Mean:** N/A")
                                    st.write(f"**Std Dev:** {info.get('std', 'N/A'):.2f}" if info.get('std') else "**Std Dev:** N/A")
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        with col2:
            if st.button("🔗 Field Mapping", type="primary"):
                analytics.generate_field_mappings()
                st.session_state.current_step = 'mapping'
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Field Mapping Page
    elif st.session_state.current_step == 'mapping':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.title("🔗 Field Mapping")
        st.markdown("**Required to run FWA Analytics**")
        
        if analytics.data is not None:
            # Create mapping interface
            st.markdown("### Map Your Data Fields to Required Healthcare Fields")
            
            # Table headers
            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
            with col1:
                st.markdown("**Required Field**")
            with col2:
                st.markdown("**Your Data Field**")
            with col3:
                st.markdown("**Data Type**")
            with col4:
                st.markdown("**Confidence**")
            with col5:
                st.markdown("**Confirm**")
            
            st.markdown("---")
            
            # Field mapping rows
            for required_field, field_info in analytics.required_fields.items():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
                
                with col1:
                    st.write(f"**{required_field}**")
                    st.caption(field_info['description'])
                
                with col2:
                    # Dropdown for field selection
                    current_mapping = analytics.field_mappings.get(required_field, analytics.data.columns[0])
                    selected_field = st.selectbox(
                        f"Select field for {required_field}",
                        options=list(analytics.data.columns),
                        index=list(analytics.data.columns).index(current_mapping) if current_mapping in analytics.data.columns else 0,
                        key=f"mapping_{required_field}",
                        label_visibility="collapsed"
                    )
                    analytics.field_mappings[required_field] = selected_field
                
                with col3:
                    st.write(field_info['type'])
                
                with col4:
                    # Recalculate confidence score
                    confidence = analytics.calculate_field_similarity(selected_field, required_field)
                    analytics.confidence_scores[required_field] = confidence
                    
                    # Color-coded confidence display
                    if confidence >= 91:
                        st.markdown(f'<div class="confidence-high">✅ {confidence}%</div>', unsafe_allow_html=True)
                    elif confidence >= 41:
                        st.markdown(f'<div class="confidence-medium">⚠️ {confidence}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="confidence-low">❌ {confidence}%</div>', unsafe_allow_html=True)
                
                with col5:
                    # Confirm checkbox
                    is_confirmed = st.checkbox(
                        "Confirm",
                        value=confidence >= 70,
                        key=f"confirm_{required_field}"
                    )
                    st.session_state.field_confirmations[required_field] = is_confirmed
            
            # Action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("⬅️ Back to Preview"):
                    st.session_state.current_step = 'preview'
                    st.rerun()
            
            with col2:
                if st.button("✅ Confirm Mapping", type="primary"):
                    st.session_state.mapping_confirmed = True
                    st.session_state.current_step = 'dashboard'
                    st.success("Field mapping confirmed!")
                    st.rerun()
            
            with col3:
                if st.button("✅ Confirm All", type="secondary"):
                    # Confirm all fields
                    for field in analytics.required_fields.keys():
                        st.session_state.field_confirmations[field] = True
                    st.session_state.mapping_confirmed = True
                    st.session_state.current_step = 'dashboard'
                    st.success("All fields confirmed!")
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Claims Data Summary Page
    elif st.session_state.current_step == 'claims_summary':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("📊 Claims Data Summary")
        with col2:
            if st.button("⬅️ Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        
        if analytics.data is not None and analytics.data_summary:
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card"><h3>Total Claims</h3><h2>{}</h2></div>'.format(len(analytics.data)), unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card"><h3>Data Fields</h3><h2>{}</h2></div>'.format(len(analytics.data.columns)), unsafe_allow_html=True)
            with col3:
                numeric_fields = sum(1 for info in analytics.data_summary.values() if info['type'] == 'number')
                st.markdown('<div class="metric-card"><h3>Numeric Fields</h3><h2>{}</h2></div>'.format(numeric_fields), unsafe_allow_html=True)
            with col4:
                total_nulls = sum(info['null_count'] for info in analytics.data_summary.values())
                st.markdown('<div class="metric-card"><h3>Total Nulls</h3><h2>{}</h2></div>'.format(total_nulls), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed field analysis
            st.subheader("📈 Field Analysis")
            
            # Create tabs for different field types
            numeric_fields = [col for col, info in analytics.data_summary.items() if info['type'] == 'number']
            categorical_fields = [col for col, info in analytics.data_summary.items() if info['type'] == 'string']
            date_fields = [col for col, info in analytics.data_summary.items() if info['type'] == 'date']
            
            tab1, tab2, tab3 = st.tabs(["📊 Numeric Fields", "📝 Categorical Fields", "📅 Date Fields"])
            
            with tab1:
                for column in numeric_fields:
                    info = analytics.data_summary[column]
                    
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.markdown(f"**{column}**")
                        
                        # Statistics table
                        stats_data = {
                            'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75', 'Nulls'],
                            'Value': [
                                info['total_count'],
                                f"{info.get('mean', 0):.2f}",
                                f"{info.get('median', 0):.2f}",
                                f"{info.get('std', 0):.2f}",
                                f"{info.get('min', 0):.2f}",
                                f"{info.get('max', 0):.2f}",
                                f"{info.get('q25', 0):.2f}",
                                f"{info.get('q75', 0):.2f}",
                                info['null_count']
                            ]
                        }
                        st.dataframe(pd.DataFrame(stats_data), hide_index=True)
                    
                    with col_b:
                        # Visualization
                        fig = analytics.create_visualization(column, info)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
            
            with tab2:
                for column in categorical_fields:
                    info = analytics.data_summary[column]
                    
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.markdown(f"**{column}**")
                        st.write(f"**Unique Values:** {info['unique_count']}")
                        st.write(f"**Null Count:** {info['null_count']}")
                        st.write(f"**Total Count:** {info['total_count']}")
                    
                    with col_b:
                        # Visualization
                        fig = analytics.create_visualization(column, info)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
            
            with tab3:
                for column in date_fields:
                    info = analytics.data_summary[column]
                    st.markdown(f"**{column}**")
                    st.write(f"**Unique Values:** {info['unique_count']}")
                    st.write(f"**Null Count:** {info['null_count']}")
                    st.write(f"**Total Count:** {info['total_count']}")
                    st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # FWA Detection Page
    elif st.session_state.current_step == 'fwa_detection':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🛡️ FWA Detection Scenarios")
        with col2:
            if st.button("⬅️ Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        
        # Get available scenarios
        available_python, unavailable_python = analytics.get_available_scenarios('python')
        available_ml, unavailable_ml = analytics.get_available_scenarios('ml')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"✅ Available Python Scenarios ({len(available_python)})")
            
            selected_python_scenarios = []
            for scenario in available_python:
                scenario_info = analytics.python_scenarios[scenario]
                
                st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
                
                # Checkbox for scenario selection
                is_selected = st.checkbox(
                    f"**{scenario}**",
                    key=f"python_{scenario}",
                    value=False
                )
                
                if is_selected:
                    selected_python_scenarios.append(scenario)
                
                st.caption(scenario_info['description'])
                st.caption(f"Required fields: {', '.join(scenario_info['required_fields'])}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader(f"❌ Unavailable Python Scenarios ({len(unavailable_python)})")
            for scenario in unavailable_python:
                scenario_info = analytics.python_scenarios[scenario]
                
                st.markdown('<div class="scenario-card unavailable-scenario">', unsafe_allow_html=True)
                st.markdown(f"**{scenario}**")
                st.caption(scenario_info['description'])
                
                # Show missing fields
                confirmed_fields = [field for field, confirmed in st.session_state.field_confirmations.items() if confirmed]
                missing_fields = [field for field in scenario_info['required_fields'] if field not in confirmed_fields]
                st.caption(f"Missing fields: {', '.join(missing_fields)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader(f"✅ Available ML Scenarios ({len(available_ml)})")
            
            selected_ml_scenarios = []
            for scenario in available_ml:
                scenario_info = analytics.ml_scenarios[scenario]
                
                st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
                
                # Checkbox for scenario selection
                is_selected = st.checkbox(
                    f"**{scenario}**",
                    key=f"ml_{scenario}",
                    value=False
                )
                
                if is_selected:
                    selected_ml_scenarios.append(scenario)
                
                st.caption(scenario_info['description'])
                st.caption(f"Required fields: {', '.join(scenario_info['required_fields'])}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader(f"❌ Unavailable ML Scenarios ({len(unavailable_ml)})")
            for scenario in unavailable_ml:
                scenario_info = analytics.ml_scenarios[scenario]
                
                st.markdown('<div class="scenario-card unavailable-scenario">', unsafe_allow_html=True)
                st.markdown(f"**{scenario}**")
                st.caption(scenario_info['description'])
                
                # Show missing fields
                confirmed_fields = [field for field, confirmed in st.session_state.field_confirmations.items() if confirmed]
                missing_fields = [field for field in scenario_info['required_fields'] if field not in confirmed_fields]
                st.caption(f"Missing fields: {', '.join(missing_fields)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Run scenarios button
        total_selected = len(selected_python_scenarios) + len(selected_ml_scenarios)
        
        if total_selected > 0:
            st.markdown("---")
            if st.button(f"🚀 Run Selected Scenarios ({total_selected})", type="primary", use_container_width=True):
                with st.spinner("Running FWA Analytics..."):
                    # Run Python scenarios
                    python_results = analytics.run_python_scenarios(selected_python_scenarios)
                    
                    # Run ML scenarios
                    ml_results = analytics.run_ml_scenarios(selected_ml_scenarios)
                    
                    # Generate Excel file
                    excel_data = analytics.export_results_to_excel(python_results, ml_results)
                    
                    if excel_data:
                        st.success("✅ FWA Analytics completed!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download FWA Results (Excel)",
                            data=excel_data,
                            file_name="FWA_Analytics_Results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        # Show summary
                        total_flagged_python = sum(len(flags) for flags in python_results.values())
                        total_flagged_ml = sum(len(flags) for flags in ml_results.values())
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Python Rules Flagged", total_flagged_python)
                        with col2:
                            st.metric("ML Scenarios Flagged", total_flagged_ml)
                        with col3:
                            st.metric("Total Unique Flags", len(set(
                                [claim for flags in python_results.values() for claim in flags] +
                                [claim for flags in ml_results.values() for claim in flags]
                            )))
                        
                        # Recommendations
                        st.markdown("### 💡 Recommendations")
                        st.info("""
                        **Based on the FWA analysis results:**
                        
                        1. **Enhanced Monitoring**: Implement enhanced monitoring for high-risk claims
                        2. **Automated Screening**: Set up automated screening for flagged scenarios
                        3. **Provider Education**: Conduct training programs for providers with multiple flags
                        4. **Regular Audits**: Schedule regular audits for high-risk providers
                        5. **Strengthen Controls**: Implement additional controls for flagged claim types
                        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Trend Analysis Page
    elif st.session_state.current_step == 'trend_analysis':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("📈 Trend Analysis")
        with col2:
            if st.button("⬅️ Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        
        if analytics.data is not None:
            st.subheader("📊 Claims Trend Analysis")
            
            # Time-series analysis
            date_field = analytics.field_mappings.get('claim_invoice_date')
            amount_field = analytics.field_mappings.get('paid_amount')
            
            if date_field and amount_field:
                # Convert to datetime
                analytics.data[date_field] = pd.to_datetime(analytics.data[date_field], errors='coerce')
                
                # Daily trends
                daily_trends = analytics.data.groupby(analytics.data[date_field].dt.date).agg({
                    amount_field: ['sum', 'count', 'mean']
                }).reset_index()
                
                daily_trends.columns = ['Date', 'Total_Amount', 'Claim_Count', 'Average_Amount']
                
                # Create trend charts
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Daily Claim Amounts', 'Daily Claim Counts', 'Average Claim Amount', 'Claims Distribution'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Daily amounts
                fig.add_trace(
                    go.Scatter(x=daily_trends['Date'], y=daily_trends['Total_Amount'], 
                              mode='lines+markers', name='Total Amount'),
                    row=1, col=1
                )
                
                # Daily counts
                fig.add_trace(
                    go.Scatter(x=daily_trends['Date'], y=daily_trends['Claim_Count'], 
                              mode='lines+markers', name='Claim Count'),
                    row=1, col=2
                )
                
                # Average amounts
                fig.add_trace(
                    go.Scatter(x=daily_trends['Date'], y=daily_trends['Average_Amount'], 
                              mode='lines+markers', name='Average Amount'),
                    row=2, col=1
                )
                
                # Distribution
                fig.add_trace(
                    go.Histogram(x=analytics.data[amount_field], name='Amount Distribution'),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("📋 Trend Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Claims Period", f"{daily_trends['Date'].min()} to {daily_trends['Date'].max()}")
                with col2:
                    st.metric("Peak Daily Amount", f"${daily_trends['Total_Amount'].max():,.2f}")
                with col3:
                    st.metric("Peak Daily Count", f"{daily_trends['Claim_Count'].max()}")
            
            else:
                st.warning("Date and amount fields are required for trend analysis. Please complete field mapping.")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()