"""
Healthcare Analytics Tool - Python Implementation
This would require: streamlit, pandas, plotly, sqlalchemy, numpy

To run locally:
pip install streamlit pandas plotly sqlalchemy numpy openpyxl
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

class HealthcareAnalytics:
    def __init__(self):
        self.data = None
        self.field_mappings = {}
        self.required_fields = {
            'claim_id': {'type': 'string', 'description': 'Unique claim identifier'},
            'patient_id': {'type': 'string', 'description': 'Patient identifier'},
            'provider_id': {'type': 'string', 'description': 'Healthcare provider ID'},
            'claim_amount': {'type': 'number', 'description': 'Total claim amount'},
            'service_date': {'type': 'date', 'description': 'Date of service'},
            'diagnosis_code': {'type': 'string', 'description': 'Primary diagnosis code'},
            'procedure_code': {'type': 'string', 'description': 'Medical procedure code'},
            'member_age': {'type': 'number', 'description': 'Patient age'},
            'service_location': {'type': 'string', 'description': 'Place of service'}
        }
        
        self.fwa_scenarios = [
            {
                'name': 'Duplicate Claims Detection',
                'required_fields': ['claim_id', 'patient_id', 'provider_id', 'claim_amount', 'service_date'],
                'description': 'Identifies potential duplicate claims submitted by providers'
            },
            {
                'name': 'Billing Pattern Analysis',
                'required_fields': ['provider_id', 'claim_amount', 'service_date', 'procedure_code'],
                'description': 'Analyzes unusual billing patterns and frequency'
            },
            {
                'name': 'Age-Service Mismatch',
                'required_fields': ['member_age', 'procedure_code', 'diagnosis_code'],
                'description': 'Detects services inappropriate for patient age'
            },
            {
                'name': 'Provider Network Analysis',
                'required_fields': ['provider_id', 'patient_id', 'service_location'],
                'description': 'Identifies suspicious provider-patient relationships'
            },
            {
                'name': 'Amount Outlier Detection',
                'required_fields': ['claim_amount', 'procedure_code', 'service_location'],
                'description': 'Detects claims with unusually high amounts'
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
            connection_string = f"{db_config['db_type']}://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            engine = create_engine(connection_string)
            
            # Sample query - you would modify this based on your database structure
            query = "SELECT * FROM claims_data LIMIT 1000"
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return None

    def generate_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data summary similar to df.describe()"""
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
        user_field = user_field.lower()
        required_field = required_field.lower()
        
        # Exact match
        if user_field == required_field:
            return 1.0
        
        # Contains match
        if required_field in user_field or user_field in required_field:
            return 0.9
        
        # Word-based similarity
        user_words = set(user_field.replace('_', ' ').split())
        required_words = set(required_field.replace('_', ' ').split())
        
        if user_words & required_words:
            return len(user_words & required_words) / len(user_words | required_words)
        
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
            
            confidence_score = int(best_score * 100)
            
            mappings.append({
                'required_field': required_field,
                'user_field': best_match or df_columns[0],
                'data_type': field_info['type'],
                'description': field_info['description'],
                'confidence_score': confidence_score,
                'is_active': confidence_score >= 70
            })
        
        return mappings

    def get_available_scenarios(self, active_mappings: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Get available and unavailable FWA scenarios based on mapped fields"""
        available = []
        unavailable = []
        
        for scenario in self.fwa_scenarios:
            if all(field in active_mappings for field in scenario['required_fields']):
                available.append(scenario)
            else:
                missing_fields = [field for field in scenario['required_fields'] if field not in active_mappings]
                scenario_copy = scenario.copy()
                scenario_copy['missing_fields'] = missing_fields
                unavailable.append(scenario_copy)
        
        return available, unavailable

    def create_visualization(self, df: pd.DataFrame, column: str):
        """Create appropriate visualization for a column"""
        if df[column].dtype in ['int64', 'float64']:
            # Histogram for numerical data
            fig = px.histogram(df, x=column, title=f'Distribution of {column}')
        else:
            # Bar chart for categorical data (top 10 values)
            value_counts = df[column].value_counts().head(10)
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f'Top 10 values in {column}')
        
        return fig

def main():
    st.set_page_config(page_title="Healthcare Analytics Tool", layout="wide")
    
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
    st.title("Healthcare Analytics Tool")
    st.markdown("Upload healthcare claims data to perform comprehensive analytics on claims datasets. Access different analytical modules including FWA (Fraud, Waste, and Abuse) analytics.")

    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        if st.button("Dashboard"):
            st.session_state.current_step = 'dashboard'
        
        if st.session_state.data_loaded:
            if st.button("Data Preview"):
                st.session_state.current_step = 'preview'
            if st.button("Field Mapping"):
                st.session_state.current_step = 'mapping'
        
        if st.session_state.mapping_confirmed:
            if st.button("Claims Summary"):
                st.session_state.current_step = 'claims_summary'
            if st.button("FWA Detection"):
                st.session_state.current_step = 'fwa_detection'
            if st.button("Trend Analysis"):
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
        uploaded_file = st.file_uploader("Upload File", type=['csv', 'xlsx', 'xls'])
        if uploaded_file:
            df = analytics.load_file_data(uploaded_file)
            if df is not None:
                analytics.data = df
                st.session_state.data_loaded = True
                st.success("File uploaded successfully!")
                st.session_state.current_step = 'preview'
                st.rerun()
    
    with col3:
        if st.button("Connect Your DB"):
            show_database_modal()

    st.markdown("---")

    # Analytical modules
    st.header("Analytics Modules")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("### 📊 Claims Data Summary")
            st.write("Comprehensive statistical summary of all available fields with visualizations and insights")
            
            if st.session_state.mapping_confirmed:
                if st.button("Open Claims Summary", key="claims_btn"):
                    st.session_state.current_step = 'claims_summary'
                    st.rerun()
            else:
                st.button("Open Claims Summary", disabled=True, help="Complete field mapping first")
    
    with col2:
        with st.container():
            st.markdown("### 🛡️ FWA Detection")
            st.write("Advanced fraud, waste, and abuse detection using multiple analytical scenarios")
            
            if st.session_state.mapping_confirmed:
                if st.button("Open FWA Detection", key="fwa_btn"):
                    st.session_state.current_step = 'fwa_detection'
                    st.rerun()
            else:
                st.button("Open FWA Detection", disabled=True, help="Complete field mapping first")
    
    with col3:
        with st.container():
            st.markdown("### 📈 Trend Analysis")
            st.write("Identify patterns and trends in healthcare claims data over time")
            
            if st.session_state.mapping_confirmed:
                if st.button("Open Trend Analysis", key="trend_btn"):
                    st.session_state.current_step = 'trend_analysis'
                    st.rerun()
            else:
                st.button("Open Trend Analysis", disabled=True, help="Complete field mapping first")

def show_database_modal():
    """Show database connection form"""
    st.subheader("Database Connection")
    
    with st.form("db_connection"):
        db_type = st.selectbox("Database Type", ["postgresql", "mysql", "sqlserver", "oracle"])
        host = st.text_input("Host", value="localhost")
        port = st.text_input("Port", value="5432")
        database = st.text_input("Database Name")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.form_submit_button("Connect"):
            db_config = {
                'db_type': db_type,
                'host': host,
                'port': port,
                'database': database,
                'username': username,
                'password': password
            }
            
            # For demo purposes, create sample data
            sample_data = {
                'claim_id': ['CLM001', 'CLM002', 'CLM003', 'CLM004', 'CLM005'],
                'patient_id': ['PAT001', 'PAT002', 'PAT003', 'PAT004', 'PAT005'],
                'provider_id': ['PROV001', 'PROV002', 'PROV001', 'PROV003', 'PROV002'],
                'claim_amount': [1500.00, 2300.50, 875.25, 3200.00, 1890.75],
                'service_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
                'diagnosis_code': ['M79.3', 'E11.9', 'I10', 'Z00.00', 'M25.511'],
                'procedure_code': ['99213', '99214', '99212', '99215', '99213'],
                'member_age': [45, 62, 38, 29, 55],
                'service_location': ['Office', 'Office', 'Office', 'Office', 'Office']
            }
            
            df = pd.DataFrame(sample_data)
            st.session_state.analytics.data = df
            st.session_state.data_loaded = True
            st.success("Connected to database successfully!")
            st.session_state.current_step = 'preview'
            st.rerun()

def show_data_preview(analytics):
    """Show data preview and summary"""
    st.header("Data Preview & Summary")
    
    if analytics.data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Preview (First 5 rows)")
            st.dataframe(analytics.data.head())
        
        with col2:
            st.subheader("Data Summary")
            summary = analytics.generate_data_summary(analytics.data)
            
            st.metric("Total Rows", summary['total_rows'])
            st.metric("Total Columns", summary['total_columns'])
            
            st.write("**Column Information:**")
            for col, info in summary['column_info'].items():
                with st.expander(f"{col} ({info['dtype']})"):
                    st.write(f"Null Count: {info['null_count']}")
                    st.write(f"Unique Values: {info['unique_count']}")
                    if 'mean' in info:
                        st.write(f"Mean: {info['mean']:.2f}")
                        st.write(f"Std: {info['std']:.2f}")
        
        if st.button("Field Mapping"):
            st.session_state.current_step = 'mapping'
            st.rerun()

def show_field_mapping(analytics):
    """Show field mapping interface"""
    st.header("Field Mapping")
    st.write("Required to run FWA Analytics")
    
    if analytics.data is not None:
        mappings = analytics.generate_field_mappings(list(analytics.data.columns))
        
        # Create form for field mappings
        with st.form("field_mapping"):
            updated_mappings = []
            
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
                        key=f"field_{i}"
                    )
                
                with col3:
                    st.write(mapping['data_type'])
                
                with col4:
                    # Confidence score with color
                    score = mapping['confidence_score']
                    if score >= 91:
                        st.success(f"{score}%")
                    elif score >= 41:
                        st.warning(f"{score}%")
                    else:
                        st.error(f"{score}%")
                
                with col5:
                    is_active = st.checkbox("Active", value=mapping['is_active'], key=f"active_{i}")
                
                updated_mapping = mapping.copy()
                updated_mapping['user_field'] = selected_field
                updated_mapping['is_active'] = is_active
                updated_mappings.append(updated_mapping)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Confirm Mapping"):
                    analytics.field_mappings = {m['required_field']: m for m in updated_mappings if m['is_active']}
                    st.session_state.mapping_confirmed = True
                    st.success("Field mapping confirmed!")
                    st.session_state.current_step = 'dashboard'
                    st.rerun()
            
            with col2:
                if st.form_submit_button("Confirm All"):
                    for mapping in updated_mappings:
                        mapping['is_active'] = True
                    analytics.field_mappings = {m['required_field']: m for m in updated_mappings}
                    st.session_state.mapping_confirmed = True
                    st.success("All fields confirmed!")
                    st.session_state.current_step = 'dashboard'
                    st.rerun()

def show_claims_summary(analytics):
    """Show comprehensive claims data summary"""
    st.header("Claims Data Summary")
    
    if analytics.data is not None:
        summary = analytics.generate_data_summary(analytics.data)
        
        # Overview metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Claims", summary['total_rows'])
        with col2:
            st.metric("Data Fields", summary['total_columns'])
        
        st.markdown("---")
        
        # Detailed field analysis
        for column, info in summary['column_info'].items():
            with st.expander(f"📊 {column} Analysis", expanded=False):
                col1, col2 = st.columns([1, 2])
                
                with col1:
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
                    try:
                        fig = analytics.create_visualization(analytics.data, column)
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.write("Visualization not available for this field")

def show_fwa_detection(analytics):
    """Show FWA detection scenarios"""
    st.header("FWA Detection Scenarios")
    
    if analytics.field_mappings:
        active_fields = list(analytics.field_mappings.keys())
        available, unavailable = analytics.get_available_scenarios(active_fields)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"✅ Available Scenarios ({len(available)})")
            
            selected_scenarios = []
            for scenario in available:
                if st.checkbox(scenario['name'], key=f"scenario_{scenario['name']}"):
                    selected_scenarios.append(scenario['name'])
                
                with st.expander(f"Details: {scenario['name']}"):
                    st.write(scenario['description'])
                    st.write(f"Required fields: {', '.join(scenario['required_fields'])}")
        
        with col2:
            st.subheader(f"❌ Unavailable Scenarios ({len(unavailable)})")
            
            for scenario in unavailable:
                st.write(f"**{scenario['name']}**")
                st.caption(scenario['description'])
                st.error(f"Missing fields: {', '.join(scenario['missing_fields'])}")
                st.markdown("---")
        
        if selected_scenarios:
            if st.button(f"Run Selected Scenarios ({len(selected_scenarios)})", type="primary"):
                st.success(f"Running FWA analysis for: {', '.join(selected_scenarios)}")
                st.info("This would process your data and generate fraud detection reports.")

def show_trend_analysis(analytics):
    """Show trend analysis module"""
    st.header("Trend Analysis")
    
    st.info("This module analyzes temporal patterns in healthcare claims data, identifying trends in claim volumes, amounts, provider behaviors, and seasonal variations.")
    
    if st.button("Start Trend Analysis", type="primary"):
        st.success("Trend analysis would be performed here with time-series visualizations and pattern detection.")

if __name__ == "__main__":
    main()