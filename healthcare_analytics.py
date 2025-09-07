import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import warnings
import sqlite3
import pymysql
import psycopg2
import pyodbc
warnings.filterwarnings('ignore')

# Import FWA scenarios module
from fwa_scenarios import PythonScenarios, MLScenarios
import json
import hashlib

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

def init_user_database():
    """Initialize SQLite database for user mappings"""
    conn = sqlite3.connect('user_mappings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            data_hash TEXT NOT NULL,
            mapping_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, data_hash)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_user_mapping(user_id, data_hash, mapping_data):
    """Save user field mapping to database"""
    try:
        conn = sqlite3.connect('user_mappings.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_mappings (user_id, data_hash, mapping_data)
            VALUES (?, ?, ?)
        ''', (user_id, data_hash, json.dumps(mapping_data)))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving mapping: {str(e)}")
        return False

def load_user_mapping(user_id, data_hash):
    """Load user field mapping from database"""
    try:
        conn = sqlite3.connect('user_mappings.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT mapping_data FROM user_mappings 
            WHERE user_id = ? AND data_hash = ?
        ''', (user_id, data_hash))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    except Exception as e:
        st.error(f"Error loading mapping: {str(e)}")
        return None

def get_user_mappings(user_id):
    """Get all mappings for a user"""
    try:
        conn = sqlite3.connect('user_mappings.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data_hash, created_at FROM user_mappings 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    except Exception as e:
        st.error(f"Error getting user mappings: {str(e)}")
        return []

def connect_to_database(db_config):
    """Connect to various database types"""
    try:
        db_type = db_config['db_type'].lower()
        
        if db_type == 'postgresql':
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['username'],
                password=db_config['password']
            )
            
        elif db_type == 'mysql':
            conn = pymysql.connect(
                host=db_config['host'],
                port=int(db_config['port']),
                database=db_config['database'],
                user=db_config['username'],
                password=db_config['password']
            )
            
        elif db_type == 'sqlite':
            conn = sqlite3.connect(db_config['database'])
            
        elif db_type == 'sqlserver':
            conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={db_config['host']},{db_config['port']};DATABASE={db_config['database']};UID={db_config['username']};PWD={db_config['password']}"
            conn = pyodbc.connect(conn_str)
            
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
            
        return conn
        
    except Exception as e:
        raise Exception(f"Database connection failed: {str(e)}")

def load_data_from_database(db_config, query=None, table_name=None):
    """Load data from database"""
    try:
        conn = connect_to_database(db_config)
        
        if query:
            df = pd.read_sql_query(query, conn)
        elif table_name:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1000", conn)
        else:
            # Get first table if no query or table specified
            if db_config['db_type'].lower() == 'postgresql':
                tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' LIMIT 1"
            elif db_config['db_type'].lower() == 'mysql':
                tables_query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{db_config['database']}' LIMIT 1"
            elif db_config['db_type'].lower() == 'sqlite':
                tables_query = "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
            else:
                tables_query = "SELECT TOP 1 TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"
                
            table_df = pd.read_sql_query(tables_query, conn)
            if not table_df.empty:
                first_table = table_df.iloc[0, 0]
                df = pd.read_sql_query(f"SELECT * FROM {first_table} LIMIT 1000", conn)
            else:
                raise Exception("No tables found in database")
        
        conn.close()
        return df
        
    except Exception as e:
        raise Exception(f"Error loading data from database: {str(e)}")

# Function to save field mappings
def save_field_mapping(data_columns, field_mappings):
    """Save field mappings based on data structure hash"""
    # Create a hash of column names for identification
    columns_hash = hashlib.md5('|'.join(sorted(data_columns)).encode()).hexdigest()
    
    mapping_data = {
        'columns_hash': columns_hash,
        'columns': data_columns,
        'mappings': field_mappings,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to session state
    if 'saved_mappings' not in st.session_state:
        st.session_state.saved_mappings = {}
    
    st.session_state.saved_mappings[columns_hash] = mapping_data
    
    # Also save to a JSON file for persistence
    try:
        with open('field_mappings.json', 'w') as f:
            json.dump(st.session_state.saved_mappings, f, indent=2)
    except:
        pass  # Ignore file save errors

# Function to load field mappings
def load_field_mapping(data_columns):
    """Load previously saved field mappings if available"""
    columns_hash = hashlib.md5('|'.join(sorted(data_columns)).encode()).hexdigest()
    
    # Try to load from session state first
    if 'saved_mappings' in st.session_state and columns_hash in st.session_state.saved_mappings:
        return st.session_state.saved_mappings[columns_hash]['mappings']
    
    # Try to load from file
    try:
        with open('field_mappings.json', 'r') as f:
            saved_mappings = json.load(f)
            if columns_hash in saved_mappings:
                st.session_state.saved_mappings = saved_mappings
                return saved_mappings[columns_hash]['mappings']
    except:
        pass  # Ignore file load errors
    
    return None

class HealthcareAnalytics:
    def __init__(self):
        # Initialize user database
        init_user_database()
        
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

    def create_dashboard_charts(self, df, selected_fields, date_range=None, version_filter=None):
        """Create dashboard charts based on selected fields"""
        charts = []
        insights = []
        
        # Filter data based on selections
        filtered_df = df.copy()
        
        if date_range and 'Treatment from date' in df.columns:
            try:
                df['Treatment from date'] = pd.to_datetime(df['Treatment from date'], errors='coerce')
                filtered_df = filtered_df[
                    (filtered_df['Treatment from date'] >= date_range[0]) & 
                    (filtered_df['Treatment from date'] <= date_range[1])
                ]
            except:
                pass
        
        if version_filter and 'Claim_version' in df.columns:
            filtered_df = filtered_df[filtered_df['Claim_version'].isin(version_filter)]
        
        for field in selected_fields:
            if field not in filtered_df.columns:
                continue
                
            # Create chart based on field type
            if pd.api.types.is_numeric_dtype(filtered_df[field]):
                # Numeric field - histogram and box plot
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=[f'{field} Distribution', f'{field} Box Plot'],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Histogram
                fig.add_trace(
                    go.Histogram(x=filtered_df[field].dropna(), name='Distribution', 
                               marker_color='rgba(102, 126, 234, 0.7)'),
                    row=1, col=1
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(y=filtered_df[field].dropna(), name='Box Plot',
                          marker_color='rgba(118, 75, 162, 0.7)'),
                    row=1, col=2
                )
                
                fig.update_layout(
                    title=f'{field} Analysis',
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                
                # Generate insights
                stats = filtered_df[field].describe()
                insight = f"""
                **{field} Insights:**
                - Mean: {stats['mean']:.2f}
                - Median: {stats['50%']:.2f}
                - Standard Deviation: {stats['std']:.2f}
                - Range: {stats['min']:.2f} to {stats['max']:.2f}
                - Outliers: {len(filtered_df[filtered_df[field] > (stats['75%'] + 1.5 * (stats['75%'] - stats['25%']))])} high outliers detected
                """
                
            else:
                # Categorical field - bar chart
                value_counts = filtered_df[field].value_counts().head(10)
                
                fig = go.Figure(data=[
                    go.Bar(x=value_counts.index, y=value_counts.values,
                          marker_color='rgba(102, 126, 234, 0.7)')
                ])
                
                fig.update_layout(
                    title=f'{field} Distribution (Top 10)',
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    xaxis_title=field,
                    yaxis_title='Count'
                )
                
                # Generate insights
                total_unique = filtered_df[field].nunique()
                most_common = value_counts.index[0]
                most_common_pct = (value_counts.iloc[0] / len(filtered_df)) * 100
                
                insight = f"""
                **{field} Insights:**
                - Total unique values: {total_unique}
                - Most common: {most_common} ({most_common_pct:.1f}% of data)
                - Distribution: {'Highly concentrated' if most_common_pct > 50 else 'Well distributed'}
                - Coverage: {(filtered_df[field].notna().sum() / len(filtered_df)) * 100:.1f}% non-null values
                """
            
            charts.append(fig)
            insights.append(insight)
        
        return charts, insights
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
    if 'current_user' not in st.session_state:
        st.session_state.current_user = 'User 1'
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    
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
    
    # Claims Data Summary with Dashboard
    elif st.session_state.current_step == 'claims-summary':
        st.markdown("## 📊 Claims Data Summary Dashboard")
        
        if st.button("← Back to Dashboard"):
            st.session_state.current_step = 'dashboard'
            st.rerun()
        
        df = st.session_state.uploaded_data
        
        # Sidebar for dashboard controls
        with st.sidebar:
            st.markdown("### 🎛️ Dashboard Controls")
            
            # Field selection
            available_fields = df.columns.tolist()
            selected_fields = st.multiselect(
                "Select Fields to Analyze",
                available_fields,
                default=available_fields[:5] if len(available_fields) >= 5 else available_fields,
                help="Choose fields for detailed analysis"
            )
            
            # Date range filter
            if 'Treatment from date' in df.columns:
                st.markdown("### 📅 Date Range")
                try:
                    df['Treatment from date'] = pd.to_datetime(df['Treatment from date'], errors='coerce')
                    min_date = df['Treatment from date'].min()
                    max_date = df['Treatment from date'].max()
                    
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                except:
                    date_range = None
            else:
                date_range = None
            
            # Version filter
            if 'Claim_version' in df.columns:
                st.markdown("### 🔢 Claim Versions")
                available_versions = sorted(df['Claim_version'].dropna().unique())
                selected_versions = st.multiselect(
                    "Select Versions",
                    available_versions,
                    default=available_versions,
                    help="Filter by claim versions"
                )
            else:
                selected_versions = None
            
            # Analysis type
            st.markdown("### 📈 Analysis Type")
            analysis_type = st.selectbox(
                "Choose Analysis",
                ["Overview", "Monthly Trends", "Version Analysis", "Field Deep Dive"]
            )
        
        # Main dashboard content
        if analysis_type == "Overview":
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Claims", f"{len(df):,}")
            
            with col2:
                if 'Paid amount' in df.columns:
                    total_amount = df['Paid amount'].sum()
                    st.metric("Total Amount", f"${total_amount:,.2f}")
                else:
                    st.metric("Total Columns", len(df.columns))
            
            with col3:
                if 'Provider ID' in df.columns:
                    unique_providers = df['Provider ID'].nunique()
                    st.metric("Unique Providers", f"{unique_providers:,}")
                else:
                    st.metric("Data Quality", f"{((df.notna().sum().sum() / (len(df) * len(df.columns))) * 100):.1f}%")
            
            with col4:
                if 'Member ID' in df.columns:
                    unique_members = df['Member ID'].nunique()
                    st.metric("Unique Members", f"{unique_members:,}")
                else:
                    st.metric("Completeness", f"{(df.dropna().shape[0] / len(df) * 100):.1f}%")
            
            # Charts for selected fields
            if selected_fields:
                charts, insights = analytics.create_dashboard_charts(
                    df, selected_fields, date_range, selected_versions
                )
                
                for i, (chart, insight) in enumerate(zip(charts, insights)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.plotly_chart(chart, use_container_width=True)
                    
                    with col2:
                        st.markdown(insight)
        
        elif analysis_type == "Monthly Trends":
            if 'Treatment from date' in df.columns:
                try:
                    df['Treatment from date'] = pd.to_datetime(df['Treatment from date'], errors='coerce')
                    df['Year_Month'] = df['Treatment from date'].dt.to_period('M')
                    
                    monthly_claims = df.groupby('Year_Month').size().reset_index(name='Claims_Count')
                    monthly_claims['Year_Month_str'] = monthly_claims['Year_Month'].astype(str)
                    
                    fig = px.line(
                        monthly_claims, 
                        x='Year_Month_str', 
                        y='Claims_Count',
                        title='Claims per Month',
                        template='plotly_dark'
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Monthly insights
                    peak_month = monthly_claims.loc[monthly_claims['Claims_Count'].idxmax(), 'Year_Month_str']
                    avg_monthly = monthly_claims['Claims_Count'].mean()
                    
                    st.markdown(f"""
                    **Monthly Trends Insights:**
                    - Peak month: {peak_month} with {monthly_claims['Claims_Count'].max():,} claims
                    - Average monthly claims: {avg_monthly:.0f}
                    - Trend: {'Increasing' if monthly_claims['Claims_Count'].iloc[-1] > avg_monthly else 'Stable/Decreasing'}
                    """)
                    
                except Exception as e:
                    st.error(f"Error creating monthly trends: {str(e)}")
            else:
                st.warning("Treatment from date field not available for monthly analysis")
        
        elif analysis_type == "Version Analysis":
            if 'Claim_version' in df.columns:
                version_counts = df['Claim_version'].value_counts().reset_index()
                version_counts.columns = ['Version', 'Count']
                
                fig = px.bar(
                    version_counts.head(10),
                    x='Version',
                    y='Count',
                    title='Claim Versions Distribution',
                    template='plotly_dark'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Version insights
                total_versions = df['Claim_version'].nunique()
                most_common_version = version_counts.iloc[0]['Version']
                most_common_pct = (version_counts.iloc[0]['Count'] / len(df)) * 100
                
                st.markdown(f"""
                **Version Analysis Insights:**
                - Total unique versions: {total_versions}
                - Most common version: {most_common_version} ({most_common_pct:.1f}% of claims)
                - Version distribution: {'Concentrated' if most_common_pct > 50 else 'Distributed'}
                """)
            else:
                st.warning("Claim_version field not available for version analysis")
        
        elif analysis_type == "Field Deep Dive":
            if selected_fields:
                field_to_analyze = st.selectbox("Select Field for Deep Analysis", selected_fields)
                
                if field_to_analyze in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Basic statistics
                        st.markdown(f"### {field_to_analyze} Statistics")
                        
                        if pd.api.types.is_numeric_dtype(df[field_to_analyze]):
                            stats = df[field_to_analyze].describe()
                            for stat, value in stats.items():
                                st.metric(stat.title(), f"{value:.2f}")
                        else:
                            st.metric("Unique Values", df[field_to_analyze].nunique())
                            st.metric("Most Common", df[field_to_analyze].mode().iloc[0] if not df[field_to_analyze].mode().empty else "N/A")
                            st.metric("Null Values", df[field_to_analyze].isnull().sum())
                    
                    with col2:
                        # Visualization
                        if pd.api.types.is_numeric_dtype(df[field_to_analyze]):
                            fig = px.histogram(
                                df, 
                                x=field_to_analyze,
                                title=f'{field_to_analyze} Distribution',
                                template='plotly_dark'
                            )
                        else:
                            value_counts = df[field_to_analyze].value_counts().head(10)
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f'{field_to_analyze} Top 10 Values',
                                template='plotly_dark'
                            )
                        
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Advanced insights
                    st.markdown(f"### {field_to_analyze} Advanced Insights")
                    
                    if pd.api.types.is_numeric_dtype(df[field_to_analyze]):
                        # Outlier analysis
                        Q1 = df[field_to_analyze].quantile(0.25)
                        Q3 = df[field_to_analyze].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = df[(df[field_to_analyze] < (Q1 - 1.5 * IQR)) | (df[field_to_analyze] > (Q3 + 1.5 * IQR))]
                        
                        st.markdown(f"""
                        **Outlier Analysis:**
                        - Total outliers: {len(outliers)} ({(len(outliers)/len(df)*100):.1f}% of data)
                        - Lower bound: {Q1 - 1.5 * IQR:.2f}
                        - Upper bound: {Q3 + 1.5 * IQR:.2f}
                        - Recommendation: {'Review high-value claims' if len(outliers) > 0 else 'Data appears normal'}
                        """)
                    else:
                        # Categorical analysis
                        value_counts = df[field_to_analyze].value_counts()
                        concentration = (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0
                        
                        st.markdown(f"""
                        **Categorical Analysis:**
                        - Data concentration: {concentration:.1f}% in top category
                        - Distribution quality: {'Poor - highly concentrated' if concentration > 70 else 'Good - well distributed' if concentration < 30 else 'Moderate concentration'}
                        - Recommendation: {'Consider data quality review' if concentration > 80 else 'Distribution appears healthy'}
                        """)
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
                    # Save mapping for current user
                    if save_user_mapping(st.session_state.current_user, st.session_state.data_hash, st.session_state.field_mappings):
                        st.success(f"💾 Field mapping saved for {st.session_state.current_user}!")
                    
                    st.session_state.mapping_confirmed = True
                    st.session_state.current_step = 'dashboard'
                    st.success("Field mapping confirmed!")
                    st.rerun()
            
            with col2:
                if st.button("✅ Confirm All Fields", use_container_width=True):
                    for mapping in st.session_state.field_mappings:
                        mapping['is_confirm'] = True
                    
                    # Save mapping for current user
                    if save_user_mapping(st.session_state.current_user, st.session_state.data_hash, st.session_state.field_mappings):
                        st.success(f"💾 Field mapping saved for {st.session_state.current_user}!")
                    
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

def run_fwa_analysis(df, selected_python, selected_ml, analytics):
    """Run FWA analysis with proper error handling"""
    try:
        with st.spinner("Running FWA Analysis..."):
            results = {
                'python_results': {},
                'ml_results': {},
                'flagged_claims': set(),
                'total_scenarios': len(selected_python) + len(selected_ml)
            }
            
            # Run Python scenarios
            if selected_python:
                st.info("Running Python-based scenarios...")
                for scenario_name in selected_python:
                    try:
                        result = analytics.python_scenarios.run_scenario(scenario_name, df)
                        if not result.empty:
                            results['python_results'][scenario_name] = result
                            # Add flagged claim IDs to set
                            if 'Claim ID' in result.columns:
                                results['flagged_claims'].update(result['Claim ID'].tolist())
                        else:
                            results['python_results'][scenario_name] = pd.DataFrame()
                    except Exception as e:
                        st.error(f"❌ Error in scenario '{scenario_name}': {str(e)}")
                        st.error("🛑 Stopping all scenarios due to error!")
                        return
            
            # Run ML scenarios with train/test split
            if selected_ml:
                st.info("Running ML-based scenarios with train/test validation...")
                for scenario_name in selected_ml:
                    try:
                        result = analytics.ml_scenarios.run_scenario(scenario_name, df)
                        if not result.empty:
                            results['ml_results'][scenario_name] = result
                            # Add flagged claim IDs to set
                            if 'Claim ID' in result.columns:
                                results['flagged_claims'].update(result['Claim ID'].tolist())
                        else:
                            results['ml_results'][scenario_name] = pd.DataFrame()
                    except Exception as e:
                        st.error(f"❌ Error in ML scenario '{scenario_name}': {str(e)}")
                        st.error("🛑 Stopping all scenarios due to error!")
                        return
            
            # Store results in session state
            st.session_state.fwa_results = results
            
            # Show summary
            total_flagged = len(results['flagged_claims'])
            st.success(f"✅ Analysis completed! {total_flagged} unique claims flagged across all scenarios.")
            
            # Create and offer download
            create_fwa_excel_report(df, results)
            
    except Exception as e:
        st.error(f"❌ Critical error during FWA analysis: {str(e)}")
        st.error("🛑 Analysis stopped!")

def create_fwa_excel_report(df, results):
    """Create Excel report with flagged claims only"""
    try:
        # Filter to only flagged claims
        flagged_claim_ids = list(results['flagged_claims'])
        
        if not flagged_claim_ids:
            st.warning("No claims were flagged by any scenario.")
            return
        
        # Filter dataframe to only flagged claims
        if 'Claim ID' in df.columns:
            flagged_df = df[df['Claim ID'].isin(flagged_claim_ids)].copy()
        else:
            st.error("Claim ID column not found in data")
            return
        
        # Create binary flags for each scenario
        python_flags = {}
        ml_flags = {}
        
        # Python scenario flags
        for scenario_name, result_df in results['python_results'].items():
            if not result_df.empty and 'Claim ID' in result_df.columns:
                flagged_ids = result_df['Claim ID'].tolist()
                python_flags[f"Python_{scenario_name.replace(' ', '_')}"] = flagged_df['Claim ID'].isin(flagged_ids).astype(int)
            else:
                python_flags[f"Python_{scenario_name.replace(' ', '_')}"] = 0
        
        # ML scenario flags
        for scenario_name, result_df in results['ml_results'].items():
            if not result_df.empty and 'Claim ID' in result_df.columns:
                flagged_ids = result_df['Claim ID'].tolist()
                ml_flags[f"ML_{scenario_name.replace(' ', '_')}"] = flagged_df['Claim ID'].isin(flagged_ids).astype(int)
            else:
                ml_flags[f"ML_{scenario_name.replace(' ', '_')}"] = 0
        
        # Calculate fraud scores
        python_score = sum(python_flags.values()) if python_flags else 0
        ml_score = sum(ml_flags.values()) if ml_flags else 0
        
        if isinstance(python_score, pd.Series):
            total_score = python_score + ml_score
        else:
            total_score = python_score + ml_score
        
        # Create Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Hypothesis sheet (Python scenarios)
            if python_flags:
                hypothesis_df = flagged_df.copy()
                for flag_name, flag_values in python_flags.items():
                    hypothesis_df[flag_name] = flag_values
                
                if isinstance(python_score, pd.Series):
                    hypothesis_df['Fraud_Score'] = python_score
                    hypothesis_df = hypothesis_df.sort_values('Fraud_Score', ascending=False)
                
                hypothesis_df.to_excel(writer, sheet_name='Hypothesis', index=False)
            
            # ML Scenarios sheet
            if ml_flags:
                ml_df = flagged_df.copy()
                for flag_name, flag_values in ml_flags.items():
                    ml_df[flag_name] = flag_values
                
                # Add ML reasoning
                ml_df['ML_Reasoning'] = 'Flagged by machine learning algorithms based on anomalous patterns in claim data'
                
                if isinstance(ml_score, pd.Series):
                    ml_df['ML_Fraud_Score'] = ml_score
                    ml_df = ml_df.sort_values('ML_Fraud_Score', ascending=False)
                
                ml_df.to_excel(writer, sheet_name='ML_Scenarios', index=False)
        
        output.seek(0)
        
        # Offer download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"FWA_Analysis_Results_{timestamp}.xlsx"
        
        st.download_button(
            label="📥 Download FWA Analysis Report",
            data=output.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success(f"✅ Report ready for download! {len(flagged_df)} flagged claims included.")
        
    except Exception as e:
        st.error(f"Error creating Excel report: {str(e)}")
        
def show_fwa_recommendations():
    """Show FWA prevention recommendations based on analysis results"""
    if 'fwa_results' not in st.session_state:
        st.error("No FWA analysis results available. Please run analysis first.")
        return
    
    results = st.session_state.fwa_results
    total_flagged = len(results['flagged_claims'])
    total_scenarios = results['total_scenarios']
    
    st.markdown("## 💡 FWA Prevention Recommendations")
    
    # Risk assessment
    risk_level = "High" if total_flagged > 100 else "Medium" if total_flagged > 50 else "Low"
    risk_color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
    
    st.markdown(f"""
    ### {risk_color} Risk Assessment: {risk_level}
    
    **Analysis Summary:**
    - Total scenarios run: {total_scenarios}
    - Unique claims flagged: {total_flagged}
    - Risk level: {risk_level}
    """)
    
    # Recommendations based on results
    recommendations = []
    
    if total_flagged > 0:
        recommendations.extend([
            "🔍 **Enhanced Monitoring**: Implement real-time monitoring for the flagged claim patterns",
            "📊 **Regular Audits**: Conduct monthly audits focusing on high-risk providers and claim types",
            "🤖 **Automated Screening**: Deploy automated pre-payment screening using the identified patterns",
            "👥 **Staff Training**: Train claims processing staff on the detected fraud indicators"
        ])
    
    if len(results['python_results']) > 0:
        recommendations.extend([
            "📋 **Rule-Based Controls**: Implement business rules based on Python scenario findings",
            "⚠️ **Alert System**: Set up alerts for claims matching the identified suspicious patterns"
        ])
    
    if len(results['ml_results']) > 0:
        recommendations.extend([
            "🧠 **ML Integration**: Integrate machine learning models into the claims processing workflow",
            "📈 **Continuous Learning**: Regularly retrain ML models with new data to improve accuracy"
        ])
    
    # General recommendations
    recommendations.extend([
        "🔐 **Access Controls**: Strengthen access controls and audit trails for claims processing",
        "📞 **Whistleblower Program**: Establish anonymous reporting channels for suspected fraud",
        "🤝 **Industry Collaboration**: Share fraud patterns with industry partners and regulatory bodies",
        "📚 **Documentation**: Maintain detailed documentation of all fraud detection activities"
    ])
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Implementation timeline
    st.markdown("""
    ### 📅 Suggested Implementation Timeline
    
    **Immediate (0-30 days):**
    - Review and investigate all flagged claims
    - Implement basic rule-based alerts
    - Train staff on new fraud indicators
    
    **Short-term (1-3 months):**
    - Deploy automated screening systems
    - Establish regular audit procedures
    - Enhance monitoring capabilities
    
    **Long-term (3-12 months):**
    - Integrate ML models into production
    - Develop comprehensive fraud prevention program
    - Establish industry partnerships
    """)

if __name__ == "__main__":
    main()