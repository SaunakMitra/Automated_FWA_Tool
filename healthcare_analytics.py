import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import FWA scenarios module
from fwa_scenarios import PythonScenarios, MLScenarios

# Page configuration
st.set_page_config(
    page_title="Healthcare FWA Analytics Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for network-inspired background and styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Network-inspired gradient background */
    .stApp {
        background: linear-gradient(135deg, 
            #1e3c72 0%, 
            #2a5298 25%, 
            #3d6bb3 50%, 
            #5084ce 75%, 
            #ff7b54 100%);
        background-attachment: fixed;
    }
    
    /* Glass morphism containers */
    .main-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Module cards */
    .module-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1.5rem;
        margin: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .module-card:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* Upload buttons */
    .upload-section {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Text styling */
    .main-title {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Confidence score styling */
    .confidence-high { color: #10b981; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #ef4444; font-weight: bold; }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

class HealthcareAnalytics:
    def __init__(self):
        self.required_fields = [
            "Claim ID", "Member ID", "Provider ID", "Provider type", "Claim invoice_no",
            "Claim_invoice line_no", "Invoice No Reference", "Claim_version", 
            "Latest_claim_version_IND", "Claim status_code", "Incident count",
            "Diagnostic_code (ICD-10)", "Procedure_code (CPT codes)", "Age", "Gender",
            "Nationality code", "Claim_invoice_date", "Admission date", "Discharge date",
            "LOS (Length_of_stay)", "POS (Place of service)", "Treatment from date",
            "Treatment to date", "Provider_country_code", "Paid amount", 
            "Claimed_currency_code", "Payment currency code", "Base currency code",
            "Claim invoice gross total amount", "Payee type", "Conversion_rate",
            "Policy_Start_&_End Dates", "Previous_Fraud_Flags", 
            "Location/Zip Code member and provider", "Coverage type (Inpatient, Outpatient, Pharmacy, etc.)",
            "Facility_type (Clinic, Hospital, Lab)", "NDC_code", "prior auth required flag",
            "prior_auth number", "prior auth approved flag", "prior_auth_approval date",
            "referral required_flag", "referral provider id", "referral submission date",
            "claim status datetime", "denial code", "denial reason", "billed_amount",
            "allowed_amount", "deductible_remaining", "copay amount", "coinsurance pct",
            "policy_code", "policy_name", "policy_type", "Policy max coverage",
            "policy_min_coverage", "Deductible Amount", "Out of Pocket Max",
            "CoPay Amount", "Coinsurance Percentage", "Policy Start Date",
            "Policy End Date or Policy Expiry Date", "Enrollment Date", "Renewal Date",
            "Premium Amount_or Monthly Premium", "Premium Frequency (e.g. monthly, quarterly)",
            "Employer Contribution", "Customer Contribution", "Discount Amount or Subsidy Amount",
            "Network Type (In-Network, Out-of-Network)", "Coverage Area or Service Area",
            "Prescription Coverage (Yes/No or details)", "Preventive Services Covered",
            "Policy Status (Active, Inactive, Cancelled)", "Is Default Policy (Boolean)",
            "Renewed_Flag"
        ]
        
        self.python_scenarios = PythonScenarios()
        self.ml_scenarios = MLScenarios()

    def load_file_data(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    def generate_data_summary(self, df):
        """Generate comprehensive data summary"""
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_info': {}
        }
        
        for column in df.columns:
            col_data = df[column].dropna()
            
            info = {
                'type': str(df[column].dtype),
                'null_count': df[column].isnull().sum(),
                'unique_count': df[column].nunique(),
                'sample_values': col_data.head(5).tolist() if len(col_data) > 0 else []
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[column]):
                info.update({
                    'mean': float(col_data.mean()) if len(col_data) > 0 else 0,
                    'median': float(col_data.median()) if len(col_data) > 0 else 0,
                    'std': float(col_data.std()) if len(col_data) > 0 else 0,
                    'min': float(col_data.min()) if len(col_data) > 0 else 0,
                    'max': float(col_data.max()) if len(col_data) > 0 else 0
                })
            
            summary['column_info'][column] = info
        
        return summary

    def generate_field_mappings(self, user_fields):
        """Generate intelligent field mappings with confidence scores"""
        mappings = []
        
        for required_field in self.required_fields:
            best_match = None
            best_score = 0
            
            for user_field in user_fields:
                score = self.calculate_similarity(user_field.lower(), required_field.lower())
                if score > best_score:
                    best_score = score
                    best_match = user_field
            
            confidence_score = int(best_score * 100)
            
            mappings.append({
                'required_field': required_field,
                'user_field': best_match if best_match else user_fields[0],
                'confidence_score': confidence_score,
                'is_confirm': confidence_score >= 70
            })
        
        return mappings

    def calculate_similarity(self, str1, str2):
        """Calculate similarity between two strings"""
        str1, str2 = str1.lower(), str2.lower()
        
        # Exact match
        if str1 == str2:
            return 1.0
        
        # Substring match
        if str1 in str2 or str2 in str1:
            return 0.9
        
        # Word-based similarity
        words1 = set(str1.replace('_', ' ').replace('-', ' ').split())
        words2 = set(str2.replace('_', ' ').replace('-', ' ').split())
        
        if words1 & words2:
            return len(words1 & words2) / len(words1 | words2)
        
        return 0.0

    def create_visualization(self, df, column):
        """Create appropriate visualization for each column type"""
        try:
            col_data = df[column].dropna()
            
            if len(col_data) == 0:
                return None
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Histogram for numeric data
                fig = px.histogram(
                    x=col_data, 
                    title=f'Distribution of {column}',
                    nbins=min(30, len(col_data.unique())),
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                return fig
            else:
                # Bar chart for categorical data
                value_counts = col_data.value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    title=f'Top 10 Values in {column}',
                    color_discrete_sequence=['#764ba2']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                return fig
        except Exception as e:
            st.error(f"Error creating visualization for {column}: {str(e)}")
            return None

    def run_fwa_analysis(self, df, selected_python_scenarios, selected_ml_scenarios):
        """Run FWA analysis with selected scenarios"""
        results = {
            'python_results': {},
            'ml_results': {},
            'flagged_claims': pd.DataFrame(),
            'error': None
        }
        
        try:
            # Run Python scenarios
            if selected_python_scenarios:
                st.info("Running Python-based FWA scenarios...")
                for scenario_name in selected_python_scenarios:
                    try:
                        result = self.python_scenarios.run_scenario(scenario_name, df)
                        results['python_results'][scenario_name] = result
                        st.success(f"✅ {scenario_name} completed")
                    except Exception as e:
                        error_msg = f"Error in {scenario_name}: {str(e)}"
                        st.error(error_msg)
                        results['error'] = error_msg
                        return results
            
            # Run ML scenarios
            if selected_ml_scenarios:
                st.info("Running ML-based FWA scenarios...")
                for scenario_name in selected_ml_scenarios:
                    try:
                        result = self.ml_scenarios.run_scenario(scenario_name, df)
                        results['ml_results'][scenario_name] = result
                        st.success(f"✅ {scenario_name} completed")
                    except Exception as e:
                        error_msg = f"Error in {scenario_name}: {str(e)}"
                        st.error(error_msg)
                        results['error'] = error_msg
                        return results
            
            # Generate flagged claims summary
            results['flagged_claims'] = self.generate_flagged_claims_summary(
                df, results['python_results'], results['ml_results']
            )
            
        except Exception as e:
            results['error'] = f"Analysis failed: {str(e)}"
        
        return results

    def generate_flagged_claims_summary(self, df, python_results, ml_results):
        """Generate summary of flagged claims with fraud scores"""
        try:
            # Create base dataframe with all claims
            flagged_df = df[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].copy()
            
            # Initialize scenario flags
            all_scenarios = list(python_results.keys()) + list(ml_results.keys())
            for scenario in all_scenarios:
                flagged_df[f'{scenario}_flag'] = 0
            
            # Mark flags for Python scenarios
            for scenario_name, result_df in python_results.items():
                if not result_df.empty and 'Claim ID' in result_df.columns:
                    flagged_claims = result_df['Claim ID'].unique()
                    flagged_df.loc[flagged_df['Claim ID'].isin(flagged_claims), f'{scenario_name}_flag'] = 1
            
            # Mark flags for ML scenarios
            for scenario_name, result_df in ml_results.items():
                if not result_df.empty and 'Claim ID' in result_df.columns:
                    flagged_claims = result_df['Claim ID'].unique()
                    flagged_df.loc[flagged_df['Claim ID'].isin(flagged_claims), f'{scenario_name}_flag'] = 1
            
            # Calculate fraud score (percentage of scenarios that flagged each claim)
            flag_columns = [col for col in flagged_df.columns if col.endswith('_flag')]
            flagged_df['fraud_score'] = (flagged_df[flag_columns].sum(axis=1) / len(flag_columns) * 100).round(2)
            
            # Sort by fraud score (highest first)
            flagged_df = flagged_df.sort_values('fraud_score', ascending=False)
            
            return flagged_df
            
        except Exception as e:
            st.error(f"Error generating flagged claims summary: {str(e)}")
            return pd.DataFrame()

    def create_excel_report(self, flagged_claims, python_results, ml_results):
        """Create Excel report with two sheets"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Hypothesis sheet (Python scenarios)
                python_flags = flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount', 'fraud_score'] + 
                                            [col for col in flagged_claims.columns if col.endswith('_flag') and 
                                             any(scenario in col for scenario in python_results.keys())]]
                python_flags.to_excel(writer, sheet_name='Hypothesis', index=False)
                
                # ML Scenarios sheet
                ml_flags = flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount', 'fraud_score'] + 
                                        [col for col in flagged_claims.columns if col.endswith('_flag') and 
                                         any(scenario in col for scenario in ml_results.keys())]]
                ml_flags.to_excel(writer, sheet_name='ML_Scenarios', index=False)
            
            return output.getvalue()
            
        except Exception as e:
            st.error(f"Error creating Excel report: {str(e)}")
            return None

def main():
    load_custom_css()
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'dashboard'
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = None
    if 'field_mappings' not in st.session_state:
        st.session_state.field_mappings = []
    if 'mapping_confirmed' not in st.session_state:
        st.session_state.mapping_confirmed = False
    if 'fwa_results' not in st.session_state:
        st.session_state.fwa_results = None
    
    analytics = HealthcareAnalytics()
    
    # Dashboard Page
    if st.session_state.current_step == 'dashboard':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # Header with upload options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown('<h1 class="main-title">Healthcare FWA Analytics Tool</h1>', unsafe_allow_html=True)
            st.markdown('<p class="subtitle">Upload healthcare claims data to perform comprehensive FWA (Fraud, Waste, and Abuse) analytics. Access different analytical modules for advanced healthcare data analysis.</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("📁 Upload File", type=['csv', 'xlsx', 'xls'])
            if st.button("🗄️ Connect Your DB"):
                st.info("Database connection feature - Connect to your healthcare database")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Process uploaded file
        if uploaded_file is not None:
            df = analytics.load_file_data(uploaded_file)
            if df is not None:
                st.session_state.uploaded_data = df
                st.session_state.data_summary = analytics.generate_data_summary(df)
                st.session_state.field_mappings = analytics.generate_field_mappings(df.columns.tolist())
                st.success(f"✅ Data loaded successfully! {len(df)} rows, {len(df.columns)} columns")
        
        # Analytics modules
        st.markdown("### Analytics Modules")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Claims Data Summary", disabled=not st.session_state.mapping_confirmed):
                st.session_state.current_step = 'claims_summary'
                st.rerun()
        
        with col2:
            if st.button("🛡️ FWA Detection", disabled=not st.session_state.mapping_confirmed):
                st.session_state.current_step = 'fwa_detection'
                st.rerun()
        
        with col3:
            if st.button("📈 Trend Analysis", disabled=not st.session_state.mapping_confirmed):
                st.session_state.current_step = 'trend_analysis'
                st.rerun()
        
        # Field mapping button
        if st.session_state.uploaded_data is not None and not st.session_state.mapping_confirmed:
            st.markdown("---")
            if st.button("🔗 Field Mapping", type="primary"):
                st.session_state.current_step = 'mapping'
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Preview Page
    elif st.session_state.current_step == 'preview':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("📋 Data Preview & Summary")
        with col2:
            if st.button("🏠 Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            
            # Data preview
            st.subheader("Data Preview (First 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data summary
            if st.session_state.data_summary:
                summary = st.session_state.data_summary
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", summary['total_rows'])
                with col2:
                    st.metric("Total Columns", summary['total_columns'])
                
                # Column information
                st.subheader("Column Information")
                for column, info in summary['column_info'].items():
                    with st.expander(f"📊 {column}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Type:** {info['type']}")
                            st.write(f"**Unique Values:** {info['unique_count']}")
                            st.write(f"**Null Count:** {info['null_count']}")
                        with col2:
                            if 'mean' in info:
                                st.write(f"**Mean:** {info['mean']:.2f}")
                                st.write(f"**Median:** {info['median']:.2f}")
                                st.write(f"**Std Dev:** {info['std']:.2f}")
        
        if st.button("🔗 Proceed to Field Mapping", type="primary"):
            st.session_state.current_step = 'mapping'
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Field Mapping Page
    elif st.session_state.current_step == 'mapping':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🔗 Field Mapping")
            st.write("Map your data fields to required healthcare analytics fields")
        with col2:
            if st.button("🏠 Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        
        if st.session_state.uploaded_data is not None and st.session_state.field_mappings:
            df = st.session_state.uploaded_data
            user_fields = df.columns.tolist()
            
            # Mapping interface
            st.subheader("Field Mapping Configuration")
            
            updated_mappings = []
            for i, mapping in enumerate(st.session_state.field_mappings):
                col1, col2, col3, col4, col5 = st.columns([3, 3, 1, 1, 1])
                
                with col1:
                    st.write(f"**{mapping['required_field']}**")
                
                with col2:
                    selected_field = st.selectbox(
                        "Select field",
                        user_fields,
                        index=user_fields.index(mapping['user_field']) if mapping['user_field'] in user_fields else 0,
                        key=f"field_{i}"
                    )
                
                with col3:
                    confidence = mapping['confidence_score']
                    if confidence >= 91:
                        st.markdown(f'<span class="confidence-high">{confidence}%</span>', unsafe_allow_html=True)
                    elif confidence >= 41:
                        st.markdown(f'<span class="confidence-medium">{confidence}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="confidence-low">{confidence}%</span>', unsafe_allow_html=True)
                
                with col4:
                    is_confirmed = st.checkbox("Confirm", value=mapping['is_confirm'], key=f"confirm_{i}")
                
                updated_mappings.append({
                    'required_field': mapping['required_field'],
                    'user_field': selected_field,
                    'confidence_score': confidence,
                    'is_confirm': is_confirmed
                })
            
            st.session_state.field_mappings = updated_mappings
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm Mapping", type="primary"):
                    st.session_state.mapping_confirmed = True
                    st.session_state.current_step = 'dashboard'
                    st.success("Field mapping confirmed!")
                    st.rerun()
            
            with col2:
                if st.button("✅ Confirm All"):
                    for mapping in st.session_state.field_mappings:
                        mapping['is_confirm'] = True
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
            if st.button("🏠 Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        
        if st.session_state.uploaded_data is not None and st.session_state.data_summary:
            df = st.session_state.uploaded_data
            summary = st.session_state.data_summary
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Claims", summary['total_rows'])
            with col2:
                st.metric("Data Fields", summary['total_columns'])
            with col3:
                if 'Paid amount' in df.columns:
                    total_amount = df['Paid amount'].sum() if pd.api.types.is_numeric_dtype(df['Paid amount']) else 0
                    st.metric("Total Paid Amount", f"${total_amount:,.2f}")
            with col4:
                if 'Provider ID' in df.columns:
                    unique_providers = df['Provider ID'].nunique()
                    st.metric("Unique Providers", unique_providers)
            
            # Visualizations for each field
            st.subheader("Field Analysis & Visualizations")
            
            # Create tabs for better organization
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
            
            tab1, tab2 = st.tabs(["📈 Numeric Fields", "📊 Categorical Fields"])
            
            with tab1:
                for column in numeric_cols[:6]:  # Show first 6 numeric columns
                    with st.expander(f"📈 {column}"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            fig = analytics.create_visualization(df, column)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            if column in summary['column_info']:
                                info = summary['column_info'][column]
                                st.write(f"**Mean:** {info.get('mean', 0):.2f}")
                                st.write(f"**Median:** {info.get('median', 0):.2f}")
                                st.write(f"**Std Dev:** {info.get('std', 0):.2f}")
                                st.write(f"**Min:** {info.get('min', 0):.2f}")
                                st.write(f"**Max:** {info.get('max', 0):.2f}")
            
            with tab2:
                for column in categorical_cols[:6]:  # Show first 6 categorical columns
                    with st.expander(f"📊 {column}"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            fig = analytics.create_visualization(df, column)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            if column in summary['column_info']:
                                info = summary['column_info'][column]
                                st.write(f"**Unique Values:** {info.get('unique_count', 0)}")
                                st.write(f"**Null Count:** {info.get('null_count', 0)}")
                                st.write(f"**Sample Values:**")
                                for val in info.get('sample_values', [])[:5]:
                                    st.write(f"• {val}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # FWA Detection Page
    elif st.session_state.current_step == 'fwa_detection':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🛡️ FWA Detection Scenarios")
        with col2:
            if st.button("🏠 Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        
        # Scenario selection with Select All/Clear All buttons
        col1, col2 = st.columns(2)
        
        # Python Scenarios
        with col1:
            st.subheader("🐍 Python-Based Scenarios")
            
            # Select All / Clear All buttons for Python scenarios
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Select All Python", key="select_all_python"):
                    for i in range(len(analytics.python_scenarios.get_available_scenarios())):
                        st.session_state[f"python_scenario_{i}"] = True
            with col_b:
                if st.button("❌ Clear All Python", key="clear_all_python"):
                    for i in range(len(analytics.python_scenarios.get_available_scenarios())):
                        st.session_state[f"python_scenario_{i}"] = False
            
            selected_python_scenarios = []
            python_scenarios = analytics.python_scenarios.get_available_scenarios()
            
            for i, (scenario_name, scenario_info) in enumerate(python_scenarios.items()):
                is_selected = st.checkbox(
                    f"**{scenario_name}**\n{scenario_info['description']}", 
                    key=f"python_scenario_{i}",
                    value=st.session_state.get(f"python_scenario_{i}", False)
                )
                if is_selected:
                    selected_python_scenarios.append(scenario_name)
        
        # ML Scenarios
        with col2:
            st.subheader("🤖 ML-Based Scenarios")
            
            # Select All / Clear All buttons for ML scenarios
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Select All ML", key="select_all_ml"):
                    for i in range(len(analytics.ml_scenarios.get_available_scenarios())):
                        st.session_state[f"ml_scenario_{i}"] = True
            with col_b:
                if st.button("❌ Clear All ML", key="clear_all_ml"):
                    for i in range(len(analytics.ml_scenarios.get_available_scenarios())):
                        st.session_state[f"ml_scenario_{i}"] = False
            
            selected_ml_scenarios = []
            ml_scenarios = analytics.ml_scenarios.get_available_scenarios()
            
            for i, (scenario_name, scenario_info) in enumerate(ml_scenarios.items()):
                is_selected = st.checkbox(
                    f"**{scenario_name}**\n{scenario_info['description']}", 
                    key=f"ml_scenario_{i}",
                    value=st.session_state.get(f"ml_scenario_{i}", False)
                )
                if is_selected:
                    selected_ml_scenarios.append(scenario_name)
        
        # Run analysis button
        st.markdown("---")
        if st.button("🚀 Run FWA Analysis", type="primary", disabled=len(selected_python_scenarios + selected_ml_scenarios) == 0):
            if st.session_state.uploaded_data is not None:
                with st.spinner("Running FWA analysis..."):
                    results = analytics.run_fwa_analysis(
                        st.session_state.uploaded_data, 
                        selected_python_scenarios, 
                        selected_ml_scenarios
                    )
                    
                    if results['error']:
                        st.error(f"❌ Analysis failed: {results['error']}")
                    else:
                        st.session_state.fwa_results = results
                        st.success("✅ FWA analysis completed successfully!")
                        
                        # Show results summary
                        flagged_claims = results['flagged_claims']
                        if not flagged_claims.empty:
                            total_flagged = (flagged_claims['fraud_score'] > 0).sum()
                            avg_fraud_score = flagged_claims['fraud_score'].mean()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Flagged Claims", total_flagged)
                            with col2:
                                st.metric("Average Fraud Score", f"{avg_fraud_score:.2f}%")
                            with col3:
                                high_risk = (flagged_claims['fraud_score'] >= 50).sum()
                                st.metric("High Risk Claims", high_risk)
                            
                            # Download Excel report
                            excel_data = analytics.create_excel_report(
                                flagged_claims, results['python_results'], results['ml_results']
                            )
                            if excel_data:
                                st.download_button(
                                    label="📥 Download FWA Analysis Report",
                                    data=excel_data,
                                    file_name=f"FWA_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Trend Analysis Page
    elif st.session_state.current_step == 'trend_analysis':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("📈 Trend Analysis")
        with col2:
            if st.button("🏠 Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            
            # Time-based analysis
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            
            if date_columns:
                selected_date_col = st.selectbox("Select Date Column for Analysis", date_columns)
                
                try:
                    # Convert to datetime
                    df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
                    df_with_dates = df.dropna(subset=[selected_date_col])
                    
                    if not df_with_dates.empty:
                        # Monthly trend analysis
                        df_with_dates['month_year'] = df_with_dates[selected_date_col].dt.to_period('M')
                        monthly_trends = df_with_dates.groupby('month_year').agg({
                            'Claim ID': 'count',
                            'Paid amount': 'sum' if 'Paid amount' in df.columns else 'count'
                        }).reset_index()
                        monthly_trends['month_year'] = monthly_trends['month_year'].astype(str)
                        
                        # Create trend charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig1 = px.line(
                                monthly_trends, 
                                x='month_year', 
                                y='Claim ID',
                                title='Claims Volume Trend',
                                color_discrete_sequence=['#667eea']
                            )
                            fig1.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='white'
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            if 'Paid amount' in df.columns:
                                fig2 = px.line(
                                    monthly_trends, 
                                    x='month_year', 
                                    y='Paid amount',
                                    title='Paid Amount Trend',
                                    color_discrete_sequence=['#764ba2']
                                )
                                fig2.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='white'
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                        
                        # Provider analysis
                        if 'Provider ID' in df.columns:
                            st.subheader("Provider Analysis")
                            provider_stats = df.groupby('Provider ID').agg({
                                'Claim ID': 'count',
                                'Paid amount': 'sum' if 'Paid amount' in df.columns else 'count'
                            }).reset_index().sort_values('Claim ID', ascending=False).head(10)
                            
                            fig3 = px.bar(
                                provider_stats, 
                                x='Provider ID', 
                                y='Claim ID',
                                title='Top 10 Providers by Claim Volume',
                                color_discrete_sequence=['#ff7b54']
                            )
                            fig3.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='white'
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error in trend analysis: {str(e)}")
            else:
                st.warning("No date columns found for trend analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()