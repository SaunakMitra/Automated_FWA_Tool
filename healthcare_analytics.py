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

# Custom CSS for network-inspired background and clean UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Network-inspired gradient background */
    .stApp {
        background: linear-gradient(135deg, 
            #1a1a2e 0%, 
            #16213e 25%, 
            #0f3460 50%, 
            #e94560 75%, 
            #f39c12 100%);
        background-attachment: fixed;
    }
    
    /* Remove extra padding and margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Glass morphism containers */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Module cards - larger and more prominent */
    .module-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 2px solid rgba(255, 255, 255, 0.4);
        padding: 2rem;
        margin: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    
    .module-card:hover {
        background: rgba(255, 255, 255, 0.95);
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.2);
        border-color: #e94560;
    }
    
    /* Upload section - compact */
    .upload-section {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Text styling with better contrast */
    .main-title {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.95);
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Button styling with better contrast */
    .stButton > button {
        background: linear-gradient(45deg, #e94560 0%, #0f3460 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #d63447 0%, #0c2d54 100%);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        color: #333;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #333 !important;
        font-weight: 500;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Confidence score styling */
    .confidence-high { 
        color: #10b981 !important; 
        font-weight: bold; 
        background: rgba(16, 185, 129, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .confidence-medium { 
        color: #f59e0b !important; 
        font-weight: bold;
        background: rgba(245, 158, 11, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .confidence-low { 
        color: #ef4444 !important; 
        font-weight: bold;
        background: rgba(239, 68, 68, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    
    /* Remove extra spacing */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact layout */
    .row-widget {
        margin-bottom: 0.5rem;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.8);
        color: #333;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #e94560, #0f3460);
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

class HealthcareAnalytics:
    def __init__(self):
        # Updated field list with new fields
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
            "Renewed_Flag", "Claim_hash_total", "Diagnostic_name", "Payee_rule_code",
            "Benefit_head_code", "Benefit_head_descr", "Country_code(Treatment Country)"
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

    def create_visualization_with_insights(self, df, column):
        """Create visualization with insights for each column"""
        try:
            col_data = df[column].dropna()
            
            if len(col_data) == 0:
                return None, "No data available for analysis"
            
            insights = []
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Histogram for numeric data
                fig = px.histogram(
                    x=col_data, 
                    title=f'Distribution of {column}',
                    nbins=min(30, len(col_data.unique())),
                    color_discrete_sequence=['#e94560']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    paper_bgcolor='rgba(255,255,255,0.9)',
                    font_color='#333',
                    title_font_color='#333',
                    title_font_size=16
                )
                
                # Generate insights
                mean_val = col_data.mean()
                median_val = col_data.median()
                std_val = col_data.std()
                
                insights.append(f"• Average {column}: ${mean_val:,.2f}" if 'amount' in column.lower() else f"• Average {column}: {mean_val:.2f}")
                insights.append(f"• Median value: ${median_val:,.2f}" if 'amount' in column.lower() else f"• Median value: {median_val:.2f}")
                
                if std_val > mean_val:
                    insights.append(f"• High variability detected (std: {std_val:.2f})")
                
                # Outlier detection
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]
                if len(outliers) > 0:
                    insights.append(f"• {len(outliers)} potential outliers detected ({len(outliers)/len(col_data)*100:.1f}%)")
                
                return fig, "\n".join(insights)
            else:
                # Bar chart for categorical data
                value_counts = col_data.value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    title=f'Top 10 Values in {column}',
                    color_discrete_sequence=['#0f3460']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    paper_bgcolor='rgba(255,255,255,0.9)',
                    font_color='#333',
                    title_font_color='#333',
                    title_font_size=16
                )
                
                # Generate insights
                total_unique = col_data.nunique()
                most_common = value_counts.index[0]
                most_common_pct = value_counts.iloc[0] / len(col_data) * 100
                
                insights.append(f"• {total_unique} unique values found")
                insights.append(f"• Most common: '{most_common}' ({most_common_pct:.1f}%)")
                
                if total_unique > len(col_data) * 0.8:
                    insights.append(f"• High diversity - most values are unique")
                elif total_unique < 10:
                    insights.append(f"• Low diversity - limited value range")
                
                return fig, "\n".join(insights)
                
        except Exception as e:
            return None, f"Error analyzing {column}: {str(e)}"

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
                st.info("🐍 Running Python-based FWA scenarios...")
                progress_bar = st.progress(0)
                
                for i, scenario_name in enumerate(selected_python_scenarios):
                    try:
                        st.write(f"Running: {scenario_name}")
                        result = self.python_scenarios.run_scenario(scenario_name, df)
                        results['python_results'][scenario_name] = result
                        progress_bar.progress((i + 1) / len(selected_python_scenarios))
                        st.success(f"✅ {scenario_name} completed")
                    except Exception as e:
                        error_msg = f"❌ Error in {scenario_name}: {str(e)}"
                        st.error(error_msg)
                        results['error'] = error_msg
                        return results
            
            # Run ML scenarios
            if selected_ml_scenarios:
                st.info("🤖 Running ML-based FWA scenarios...")
                progress_bar = st.progress(0)
                
                for i, scenario_name in enumerate(selected_ml_scenarios):
                    try:
                        st.write(f"Running: {scenario_name}")
                        result = self.ml_scenarios.run_scenario(scenario_name, df)
                        results['ml_results'][scenario_name] = result
                        progress_bar.progress((i + 1) / len(selected_ml_scenarios))
                        st.success(f"✅ {scenario_name} completed")
                    except Exception as e:
                        error_msg = f"❌ Error in {scenario_name}: {str(e)}"
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
            # Get all flagged claim IDs
            all_flagged_claim_ids = set()
            
            # Collect flagged claims from Python scenarios
            for scenario_name, result_df in python_results.items():
                if not result_df.empty and 'Claim ID' in result_df.columns:
                    all_flagged_claim_ids.update(result_df['Claim ID'].unique())
            
            # Collect flagged claims from ML scenarios
            for scenario_name, result_df in ml_results.items():
                if not result_df.empty and 'Claim ID' in result_df.columns:
                    all_flagged_claim_ids.update(result_df['Claim ID'].unique())
            
            if not all_flagged_claim_ids:
                return pd.DataFrame()
            
            # Create base dataframe with only flagged claims
            flagged_df = df[df['Claim ID'].isin(all_flagged_claim_ids)][['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].copy()
            
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

    def generate_fwa_recommendations(self, flagged_claims, total_claims):
        """Generate FWA prevention recommendations based on analysis"""
        recommendations = []
        
        if flagged_claims.empty:
            return ["✅ No suspicious claims detected. Continue monitoring with current controls."]
        
        fraud_rate = len(flagged_claims) / total_claims * 100
        high_risk_claims = (flagged_claims['fraud_score'] >= 70).sum()
        
        # Risk level assessment
        if fraud_rate > 10:
            recommendations.append("🚨 **HIGH RISK**: Fraud rate exceeds 10%. Immediate action required.")
        elif fraud_rate > 5:
            recommendations.append("⚠️ **MEDIUM RISK**: Fraud rate between 5-10%. Enhanced monitoring needed.")
        else:
            recommendations.append("✅ **LOW RISK**: Fraud rate below 5%. Maintain current controls.")
        
        # Specific recommendations
        recommendations.extend([
            f"📊 **Analysis Summary**: {len(flagged_claims)} claims flagged out of {total_claims} total claims ({fraud_rate:.2f}%)",
            f"🎯 **High Priority**: {high_risk_claims} claims with fraud score ≥70% require immediate review",
            "",
            "🛡️ **Recommended Actions:**",
            "• Implement automated pre-authorization for high-risk claims",
            "• Enhance provider credentialing and monitoring processes",
            "• Deploy real-time claim validation rules",
            "• Establish regular audit schedules for flagged providers",
            "• Implement member education programs on appropriate healthcare utilization",
            "",
            "📈 **Process Improvements:**",
            "• Set up automated alerts for claims exceeding fraud score thresholds",
            "• Create provider scorecards based on historical fraud patterns",
            "• Implement geographic validation for treatment locations",
            "• Enhance duplicate claim detection algorithms",
            "",
            "🔍 **Monitoring Enhancements:**",
            "• Weekly review of high-risk provider patterns",
            "• Monthly trend analysis of fraud indicators",
            "• Quarterly assessment of control effectiveness",
            "• Annual review and update of detection scenarios"
        ])
        
        return recommendations

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
        # Header with upload options - compact layout
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown('<h1 class="main-title">Healthcare FWA Analytics Tool</h1>', unsafe_allow_html=True)
            st.markdown('<p class="subtitle">Upload healthcare claims data to perform comprehensive FWA (Fraud, Waste, and Abuse) analytics with advanced detection algorithms.</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("📁 Upload File", type=['csv', 'xlsx', 'xls'], label_visibility="collapsed")
            if st.button("🗄️ Connect DB", use_container_width=True):
                st.info("Database connection feature available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Process uploaded file
        if uploaded_file is not None:
            df = analytics.load_file_data(uploaded_file)
            if df is not None:
                st.session_state.uploaded_data = df
                st.session_state.data_summary = analytics.generate_data_summary(df)
                st.session_state.field_mappings = analytics.generate_field_mappings(df.columns.tolist())
                st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Analytics modules - larger and more prominent
        st.markdown("### Analytics Modules")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="module-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Claims Data Summary")
            st.markdown("Comprehensive statistical analysis with visualizations and insights")
            if st.button("Open Claims Summary", disabled=not st.session_state.mapping_confirmed, use_container_width=True):
                st.session_state.current_step = 'claims_summary'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="module-card">', unsafe_allow_html=True)
            st.markdown("### 🛡️ FWA Detection")
            st.markdown("Advanced fraud, waste, and abuse detection using Python rules and ML algorithms")
            if st.button("Open FWA Detection", disabled=not st.session_state.mapping_confirmed, use_container_width=True):
                st.session_state.current_step = 'fwa_detection'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="module-card">', unsafe_allow_html=True)
            st.markdown("### 📈 Trend Analysis")
            st.markdown("Time-series analysis of claim patterns and provider behaviors")
            if st.button("Open Trend Analysis", disabled=not st.session_state.mapping_confirmed, use_container_width=True):
                st.session_state.current_step = 'trend_analysis'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Field mapping button
        if st.session_state.uploaded_data is not None and not st.session_state.mapping_confirmed:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔗 Start Field Mapping", type="primary", use_container_width=True):
                    st.session_state.current_step = 'mapping'
                    st.rerun()
    
    # Field Mapping Page - Compact Layout
    elif st.session_state.current_step == 'mapping':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
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
            
            # Compact mapping interface - show 10 at a time
            st.subheader("Field Mapping Configuration")
            
            # Pagination for field mapping
            items_per_page = 15
            total_fields = len(st.session_state.field_mappings)
            total_pages = (total_fields + items_per_page - 1) // items_per_page
            
            if 'mapping_page' not in st.session_state:
                st.session_state.mapping_page = 0
            
            # Page navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", disabled=st.session_state.mapping_page == 0):
                    st.session_state.mapping_page -= 1
                    st.rerun()
            with col2:
                st.write(f"Page {st.session_state.mapping_page + 1} of {total_pages}")
            with col3:
                if st.button("➡️ Next", disabled=st.session_state.mapping_page >= total_pages - 1):
                    st.session_state.mapping_page += 1
                    st.rerun()
            
            # Show current page mappings
            start_idx = st.session_state.mapping_page * items_per_page
            end_idx = min(start_idx + items_per_page, total_fields)
            
            updated_mappings = st.session_state.field_mappings.copy()
            
            for i in range(start_idx, end_idx):
                mapping = st.session_state.field_mappings[i]
                col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
                
                with col1:
                    st.write(f"**{mapping['required_field']}**")
                
                with col2:
                    selected_field = st.selectbox(
                        "Select field",
                        user_fields,
                        index=user_fields.index(mapping['user_field']) if mapping['user_field'] in user_fields else 0,
                        key=f"field_{i}",
                        label_visibility="collapsed"
                    )
                    updated_mappings[i]['user_field'] = selected_field
                
                with col3:
                    confidence = mapping['confidence_score']
                    if confidence >= 70:
                        st.markdown(f'<span class="confidence-high">{confidence}%</span>', unsafe_allow_html=True)
                    elif confidence >= 40:
                        st.markdown(f'<span class="confidence-medium">{confidence}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="confidence-low">{confidence}%</span>', unsafe_allow_html=True)
                
                with col4:
                    is_confirmed = st.checkbox("Confirm", value=mapping['is_confirm'], key=f"confirm_{i}")
                    updated_mappings[i]['is_confirm'] = is_confirmed
            
            st.session_state.field_mappings = updated_mappings
            
            # Action buttons
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm Mapping", type="primary", use_container_width=True):
                    st.session_state.mapping_confirmed = True
                    st.session_state.current_step = 'dashboard'
                    st.success("Field mapping confirmed!")
                    st.rerun()
            
            with col2:
                if st.button("✅ Confirm All Fields", use_container_width=True):
                    for mapping in st.session_state.field_mappings:
                        mapping['is_confirm'] = True
                    st.session_state.mapping_confirmed = True
                    st.session_state.current_step = 'dashboard'
                    st.success("All fields confirmed!")
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Claims Data Summary Page - Enhanced
    elif st.session_state.current_step == 'claims_summary':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.title("📊 Claims Data Summary")
        with col2:
            if st.button("🏠 Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            
            # Key metrics in one row
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Claims", len(df))
            with col2:
                st.metric("Data Fields", len(df.columns))
            with col3:
                if 'Paid amount' in df.columns:
                    total_amount = pd.to_numeric(df['Paid amount'], errors='coerce').sum()
                    st.metric("Total Paid", f"${total_amount:,.0f}")
            with col4:
                if 'Provider ID' in df.columns:
                    st.metric("Unique Providers", df['Provider ID'].nunique())
            with col5:
                if 'Member ID' in df.columns:
                    st.metric("Unique Members", df['Member ID'].nunique())
            
            # Claims per month/year analysis
            st.subheader("📅 Claims Timeline Analysis")
            
            date_columns = [col for col in df.columns if 'date' in col.lower() and 'from' in col.lower()]
            if date_columns:
                selected_date_col = st.selectbox("Select Date Column", date_columns)
                
                try:
                    df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
                    df_with_dates = df.dropna(subset=[selected_date_col])
                    
                    if not df_with_dates.empty:
                        # Monthly and yearly analysis
                        df_with_dates['month_year'] = df_with_dates[selected_date_col].dt.to_period('M')
                        df_with_dates['year'] = df_with_dates[selected_date_col].dt.year
                        
                        monthly_claims = df_with_dates.groupby('month_year').size().reset_index(name='claim_count')
                        yearly_claims = df_with_dates.groupby('year').size().reset_index(name='claim_count')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Monthly claims chart
                            monthly_claims['month_year_str'] = monthly_claims['month_year'].astype(str)
                            fig1 = px.line(monthly_claims, x='month_year_str', y='claim_count',
                                         title='Claims per Month', markers=True)
                            fig1.update_layout(
                                plot_bgcolor='rgba(255,255,255,0.9)',
                                paper_bgcolor='rgba(255,255,255,0.9)',
                                font_color='#333'
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            # Yearly claims chart
                            fig2 = px.bar(yearly_claims, x='year', y='claim_count',
                                        title='Claims per Year')
                            fig2.update_layout(
                                plot_bgcolor='rgba(255,255,255,0.9)',
                                paper_bgcolor='rgba(255,255,255,0.9)',
                                font_color='#333'
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Version analysis
                        if 'Claim_version' in df.columns:
                            st.subheader("📋 Claim Versions Analysis")
                            version_counts = df['Claim_version'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Versions", len(version_counts))
                                st.write("**Version Distribution:**")
                                for version, count in version_counts.head(5).items():
                                    st.write(f"• Version {version}: {count} claims")
                            
                            with col2:
                                fig3 = px.pie(values=version_counts.values, names=version_counts.index,
                                            title='Claim Version Distribution')
                                fig3.update_layout(
                                    plot_bgcolor='rgba(255,255,255,0.9)',
                                    paper_bgcolor='rgba(255,255,255,0.9)',
                                    font_color='#333'
                                )
                                st.plotly_chart(fig3, use_container_width=True)
                        
                        # Month/Year dropdown filter
                        st.subheader("🔍 Detailed Analysis by Period")
                        period_type = st.selectbox("Select Period Type", ["Month", "Year"])
                        
                        if period_type == "Month":
                            selected_period = st.selectbox("Select Month", monthly_claims['month_year_str'].unique())
                            filtered_data = df_with_dates[df_with_dates['month_year'].astype(str) == selected_period]
                        else:
                            selected_period = st.selectbox("Select Year", yearly_claims['year'].unique())
                            filtered_data = df_with_dates[df_with_dates['year'] == selected_period]
                        
                        if not filtered_data.empty:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Claims in Period", len(filtered_data))
                            with col2:
                                if 'Paid amount' in filtered_data.columns:
                                    period_amount = pd.to_numeric(filtered_data['Paid amount'], errors='coerce').sum()
                                    st.metric("Total Paid", f"${period_amount:,.0f}")
                            with col3:
                                if 'Provider ID' in filtered_data.columns:
                                    st.metric("Active Providers", filtered_data['Provider ID'].nunique())
                
                except Exception as e:
                    st.error(f"Error in timeline analysis: {str(e)}")
            
            # Field-specific analysis with dropdown
            st.subheader("🔍 Field-Specific Analysis")
            
            # Dropdown for field selection
            available_fields = [col for col in df.columns if col in ['Paid amount', 'Age', 'Provider ID', 'Member ID', 'Diagnostic_code (ICD-10)', 'Procedure_code (CPT codes)']]
            if available_fields:
                selected_field = st.selectbox("Select Field for Analysis", available_fields)
                
                if selected_field:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig, insights = analytics.create_visualization_with_insights(df, selected_field)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.markdown("**📋 Analysis Insights:**")
                        st.markdown(insights)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # FWA Detection Page - Enhanced with Dropdowns
    elif st.session_state.current_step == 'fwa_detection':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.title("🛡️ FWA Detection Scenarios")
        with col2:
            if st.button("🏠 Back to Dashboard"):
                st.session_state.current_step = 'dashboard'
                st.rerun()
        
        # Scenario selection with dropdowns
        col1, col2 = st.columns(2)
        
        # Python Scenarios
        with col1:
            st.subheader("🐍 Python-Based Scenarios")
            
            # Select All / Clear All buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Select All Python", key="select_all_python", use_container_width=True):
                    python_scenarios = analytics.python_scenarios.get_available_scenarios()
                    for i in range(len(python_scenarios)):
                        st.session_state[f"python_scenario_{i}"] = True
                    st.rerun()
            with col_b:
                if st.button("❌ Clear All Python", key="clear_all_python", use_container_width=True):
                    python_scenarios = analytics.python_scenarios.get_available_scenarios()
                    for i in range(len(python_scenarios)):
                        st.session_state[f"python_scenario_{i}"] = False
                    st.rerun()
            
            # Dropdown for Python scenarios
            python_scenarios = analytics.python_scenarios.get_available_scenarios()
            selected_python_scenarios = st.multiselect(
                "Select Python Scenarios",
                list(python_scenarios.keys()),
                default=[name for i, name in enumerate(python_scenarios.keys()) 
                        if st.session_state.get(f"python_scenario_{i}", False)]
            )
            
            # Update session state based on multiselect
            for i, scenario_name in enumerate(python_scenarios.keys()):
                st.session_state[f"python_scenario_{i}"] = scenario_name in selected_python_scenarios
        
        # ML Scenarios
        with col2:
            st.subheader("🤖 ML-Based Scenarios")
            
            # Select All / Clear All buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Select All ML", key="select_all_ml", use_container_width=True):
                    ml_scenarios = analytics.ml_scenarios.get_available_scenarios()
                    for i in range(len(ml_scenarios)):
                        st.session_state[f"ml_scenario_{i}"] = True
                    st.rerun()
            with col_b:
                if st.button("❌ Clear All ML", key="clear_all_ml", use_container_width=True):
                    ml_scenarios = analytics.ml_scenarios.get_available_scenarios()
                    for i in range(len(ml_scenarios)):
                        st.session_state[f"ml_scenario_{i}"] = False
                    st.rerun()
            
            # Dropdown for ML scenarios
            ml_scenarios = analytics.ml_scenarios.get_available_scenarios()
            selected_ml_scenarios = st.multiselect(
                "Select ML Scenarios",
                list(ml_scenarios.keys()),
                default=[name for i, name in enumerate(ml_scenarios.keys()) 
                        if st.session_state.get(f"ml_scenario_{i}", False)]
            )
            
            # Update session state based on multiselect
            for i, scenario_name in enumerate(ml_scenarios.keys()):
                st.session_state[f"ml_scenario_{i}"] = scenario_name in selected_ml_scenarios
        
        # Run analysis button
        st.markdown("---")
        total_selected = len(selected_python_scenarios) + len(selected_ml_scenarios)
        
        if st.button(f"🚀 Run FWA Analysis ({total_selected} scenarios selected)", 
                    type="primary", disabled=total_selected == 0, use_container_width=True):
            if st.session_state.uploaded_data is not None:
                with st.spinner("Running comprehensive FWA analysis..."):
                    results = analytics.run_fwa_analysis(
                        st.session_state.uploaded_data, 
                        selected_python_scenarios, 
                        selected_ml_scenarios
                    )
                    
                    if results['error']:
                        st.error(f"❌ Analysis failed: {results['error']}")
                        st.error("All scenarios stopped due to error. Please check your data and field mappings.")
                    else:
                        st.session_state.fwa_results = results
                        st.success("✅ FWA analysis completed successfully!")
                        
                        # Show results summary
                        flagged_claims = results['flagged_claims']
                        if not flagged_claims.empty:
                            total_flagged = len(flagged_claims)
                            avg_fraud_score = flagged_claims['fraud_score'].mean()
                            high_risk = (flagged_claims['fraud_score'] >= 70).sum()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Flagged Claims", total_flagged)
                            with col2:
                                st.metric("Avg Fraud Score", f"{avg_fraud_score:.1f}%")
                            with col3:
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
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            # FWA Prevention Recommendations
                            st.markdown("---")
                            if st.button("💡 Get FWA Prevention Recommendations", use_container_width=True):
                                recommendations = analytics.generate_fwa_recommendations(
                                    flagged_claims, len(st.session_state.uploaded_data)
                                )
                                
                                st.subheader("🛡️ FWA Prevention Recommendations")
                                for rec in recommendations:
                                    st.markdown(rec)
                        else:
                            st.info("No suspicious claims detected in the analysis.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Trend Analysis Page - Enhanced and Interactive
    elif st.session_state.current_step == 'trend_analysis':
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
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
                        # Enhanced trend analysis
                        df_with_dates['month_year'] = df_with_dates[selected_date_col].dt.to_period('M')
                        df_with_dates['quarter'] = df_with_dates[selected_date_col].dt.to_period('Q')
                        df_with_dates['year'] = df_with_dates[selected_date_col].dt.year
                        df_with_dates['day_of_week'] = df_with_dates[selected_date_col].dt.day_name()
                        
                        # Multiple trend analyses
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Claims volume trend
                            monthly_trends = df_with_dates.groupby('month_year').agg({
                                'Claim ID': 'count',
                                'Paid amount': 'sum' if 'Paid amount' in df.columns else 'count'
                            }).reset_index()
                            monthly_trends['month_year'] = monthly_trends['month_year'].astype(str)
                            
                            fig1 = px.line(monthly_trends, x='month_year', y='Claim ID',
                                         title='Claims Volume Trend Over Time', markers=True)
                            fig1.update_layout(
                                plot_bgcolor='rgba(255,255,255,0.9)',
                                paper_bgcolor='rgba(255,255,255,0.9)',
                                font_color='#333'
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            # Day of week analysis
                            dow_analysis = df_with_dates.groupby('day_of_week').size().reset_index(name='claim_count')
                            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            dow_analysis['day_of_week'] = pd.Categorical(dow_analysis['day_of_week'], categories=dow_order, ordered=True)
                            dow_analysis = dow_analysis.sort_values('day_of_week')
                            
                            fig2 = px.bar(dow_analysis, x='day_of_week', y='claim_count',
                                        title='Claims by Day of Week')
                            fig2.update_layout(
                                plot_bgcolor='rgba(255,255,255,0.9)',
                                paper_bgcolor='rgba(255,255,255,0.9)',
                                font_color='#333'
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Provider and Member analysis
                        if 'Provider ID' in df.columns:
                            st.subheader("🏥 Provider Analysis")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                # Top providers by volume
                                provider_stats = df.groupby('Provider ID').agg({
                                    'Claim ID': 'count',
                                    'Paid amount': 'sum' if 'Paid amount' in df.columns else 'count'
                                }).reset_index().sort_values('Claim ID', ascending=False).head(10)
                                
                                fig3 = px.bar(provider_stats, x='Provider ID', y='Claim ID',
                                            title='Top 10 Providers by Claim Volume')
                                fig3.update_layout(
                                    plot_bgcolor='rgba(255,255,255,0.9)',
                                    paper_bgcolor='rgba(255,255,255,0.9)',
                                    font_color='#333'
                                )
                                st.plotly_chart(fig3, use_container_width=True)
                            
                            with col2:
                                # Provider distribution analysis
                                provider_claim_counts = df.groupby('Provider ID').size()
                                
                                # Categorize providers
                                low_volume = (provider_claim_counts <= 5).sum()
                                medium_volume = ((provider_claim_counts > 5) & (provider_claim_counts <= 20)).sum()
                                high_volume = (provider_claim_counts > 20).sum()
                                
                                provider_categories = pd.DataFrame({
                                    'Category': ['Low Volume (≤5)', 'Medium Volume (6-20)', 'High Volume (>20)'],
                                    'Count': [low_volume, medium_volume, high_volume]
                                })
                                
                                fig4 = px.pie(provider_categories, values='Count', names='Category',
                                            title='Provider Volume Distribution')
                                fig4.update_layout(
                                    plot_bgcolor='rgba(255,255,255,0.9)',
                                    paper_bgcolor='rgba(255,255,255,0.9)',
                                    font_color='#333'
                                )
                                st.plotly_chart(fig4, use_container_width=True)
                        
                        # Key insights
                        st.subheader("🔍 Key Findings")
                        
                        insights = []
                        
                        # Seasonal patterns
                        if len(monthly_trends) > 3:
                            max_month = monthly_trends.loc[monthly_trends['Claim ID'].idxmax(), 'month_year']
                            min_month = monthly_trends.loc[monthly_trends['Claim ID'].idxmin(), 'month_year']
                            insights.append(f"📈 Peak claims month: {max_month}")
                            insights.append(f"📉 Lowest claims month: {min_month}")
                        
                        # Weekend vs weekday patterns
                        weekend_claims = df_with_dates[df_with_dates[selected_date_col].dt.dayofweek.isin([5, 6])].shape[0]
                        weekday_claims = df_with_dates[~df_with_dates[selected_date_col].dt.dayofweek.isin([5, 6])].shape[0]
                        weekend_pct = weekend_claims / (weekend_claims + weekday_claims) * 100
                        
                        if weekend_pct > 15:
                            insights.append(f"⚠️ High weekend activity: {weekend_pct:.1f}% of claims on weekends")
                        else:
                            insights.append(f"✅ Normal weekend activity: {weekend_pct:.1f}% of claims on weekends")
                        
                        # Provider concentration
                        if 'Provider ID' in df.columns:
                            top_10_providers_pct = provider_stats.head(10)['Claim ID'].sum() / len(df) * 100
                            insights.append(f"🏥 Top 10 providers handle {top_10_providers_pct:.1f}% of all claims")
                        
                        for insight in insights:
                            st.markdown(f"• {insight}")
                
                except Exception as e:
                    st.error(f"Error in trend analysis: {str(e)}")
            else:
                st.warning("No date columns found for trend analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()