import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Database connection libraries
try:
    import psycopg2
    import pymysql
    import pyodbc
except ImportError:
    pass

# ML and forecasting libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import lightgbm as lgb

# Forecasting libraries
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import FWA scenarios
from fwa_scenarios import FWAScenarios

# Page configuration
st.set_page_config(
    page_title="Healthcare Analytics Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Power BI style dashboard
def load_custom_css():
    st.markdown("""
    <style>
    /* Main background with network pattern */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Remove default padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Dashboard cards */
    .dashboard-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Remove empty spaces */
    .element-container:has(> .stEmpty) {
        display: none;
    }
    
    /* High contrast text */
    .stMarkdown, .stText {
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dashboard grid */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

class HealthcareAnalytics:
    def __init__(self):
        self.init_database()
        self.required_fields = {
            "Claim ID": {"type": "string", "description": "Unique claim identifier"},
            "Member ID": {"type": "string", "description": "Patient identifier"},
            "Provider ID": {"type": "string", "description": "Healthcare provider ID"},
            "Provider type": {"type": "string", "description": "Type of healthcare provider"},
            "Claim invoice_no": {"type": "string", "description": "Invoice number"},
            "Claim_invoice line_no": {"type": "string", "description": "Invoice line number"},
            "Invoice No Reference": {"type": "string", "description": "Invoice reference number"},
            "Claim_version": {"type": "string", "description": "Version of the claim"},
            "Latest_claim_version_IND": {"type": "string", "description": "Latest version indicator"},
            "Claim status_code": {"type": "string", "description": "Status code of claim"},
            "Incident count": {"type": "number", "description": "Number of incidents"},
            "Diagnostic_code (ICD-10)": {"type": "string", "description": "ICD-10 diagnostic code"},
            "Procedure_code (CPT codes)": {"type": "string", "description": "CPT procedure code"},
            "Age": {"type": "number", "description": "Patient age"},
            "Gender": {"type": "string", "description": "Patient gender"},
            "Nationality code": {"type": "string", "description": "Patient nationality"},
            "Claim_invoice_date": {"type": "date", "description": "Invoice date"},
            "Admission date": {"type": "date", "description": "Hospital admission date"},
            "Discharge date": {"type": "date", "description": "Hospital discharge date"},
            "LOS (Length_of_stay)": {"type": "number", "description": "Length of stay"},
            "POS (Place of service)": {"type": "string", "description": "Place of service"},
            "Treatment from date": {"type": "date", "description": "Treatment start date"},
            "Treatment to date": {"type": "date", "description": "Treatment end date"},
            "Provider_country_code": {"type": "string", "description": "Provider country"},
            "Paid amount": {"type": "number", "description": "Amount paid"},
            "Claimed_currency_code": {"type": "string", "description": "Claimed currency"},
            "Payment currency code": {"type": "string", "description": "Payment currency"},
            "Base currency code": {"type": "string", "description": "Base currency"},
            "Claim invoice gross total amount": {"type": "number", "description": "Gross total amount"},
            "Payee type": {"type": "string", "description": "Type of payee"},
            "Conversion_rate": {"type": "number", "description": "Currency conversion rate"},
            "Policy_Start_&_End Dates": {"type": "date", "description": "Policy period"},
            "Previous_Fraud_Flags": {"type": "string", "description": "Previous fraud indicators"},
            "Location/Zip Code member and provider": {"type": "string", "description": "Geographic location"},
            "Coverage type (Inpatient, Outpatient, Pharmacy, etc.)": {"type": "string", "description": "Type of coverage"},
            "Facility_type (Clinic, Hospital, Lab)": {"type": "string", "description": "Healthcare facility type"},
            "NDC_code": {"type": "string", "description": "National Drug Code"},
            "prior auth required flag": {"type": "string", "description": "Prior authorization required"},
            "prior_auth number": {"type": "string", "description": "Prior authorization number"},
            "prior auth approved flag": {"type": "string", "description": "Prior auth approval status"},
            "prior_auth_approval date": {"type": "date", "description": "Prior auth approval date"},
            "referral required_flag": {"type": "string", "description": "Referral requirement flag"},
            "referral provider id": {"type": "string", "description": "Referring provider ID"},
            "referral submission date": {"type": "date", "description": "Referral submission date"},
            "claim status datetime": {"type": "date", "description": "Claim status timestamp"},
            "denial code": {"type": "string", "description": "Denial code if applicable"},
            "denial reason": {"type": "string", "description": "Reason for denial"},
            "billed_amount": {"type": "number", "description": "Originally billed amount"},
            "allowed_amount": {"type": "number", "description": "Allowed amount"},
            "deductible_remaining": {"type": "number", "description": "Remaining deductible"},
            "copay amount": {"type": "number", "description": "Copayment amount"},
            "coinsurance pct": {"type": "number", "description": "Coinsurance percentage"},
            "policy_code": {"type": "string", "description": "Policy code"},
            "policy_name": {"type": "string", "description": "Policy name"},
            "policy_type": {"type": "string", "description": "Type of policy"},
            "Policy max coverage": {"type": "number", "description": "Maximum coverage amount"},
            "policy_min_coverage": {"type": "number", "description": "Minimum coverage amount"},
            "Deductible Amount": {"type": "number", "description": "Deductible amount"},
            "Out of Pocket Max": {"type": "number", "description": "Out-of-pocket maximum"},
            "CoPay Amount": {"type": "number", "description": "Copay amount"},
            "Coinsurance Percentage": {"type": "number", "description": "Coinsurance percentage"},
            "Policy Start Date": {"type": "date", "description": "Policy start date"},
            "Policy End Date or Policy Expiry Date": {"type": "date", "description": "Policy end date"},
            "Enrollment Date": {"type": "date", "description": "Enrollment date"},
            "Renewal Date": {"type": "date", "description": "Policy renewal date"},
            "Premium Amount_or Monthly Premium": {"type": "number", "description": "Premium amount"},
            "Premium Frequency (e.g. monthly, quarterly)": {"type": "string", "description": "Premium frequency"},
            "Employer Contribution": {"type": "number", "description": "Employer contribution"},
            "Customer Contribution": {"type": "number", "description": "Customer contribution"},
            "Discount Amount or Subsidy Amount": {"type": "number", "description": "Discount amount"},
            "Network Type (In-Network, Out-of-Network)": {"type": "string", "description": "Network type"},
            "Coverage Area or Service Area": {"type": "string", "description": "Coverage area"},
            "Prescription Coverage (Yes/No or details)": {"type": "string", "description": "Prescription coverage"},
            "Preventive Services Covered": {"type": "string", "description": "Preventive services coverage"},
            "Policy Status (Active, Inactive, Cancelled)": {"type": "string", "description": "Policy status"},
            "Is Default Policy (Boolean)": {"type": "string", "description": "Default policy indicator"},
            "Renewed_Flag": {"type": "string", "description": "Renewal flag"},
            "Claim_hash_total": {"type": "string", "description": "Claim hash total"},
            "Diagnostic_name": {"type": "string", "description": "Diagnostic name"},
            "Payee_rule_code": {"type": "string", "description": "Payee rule code"},
            "Benefit_head_code": {"type": "string", "description": "Benefit head code"},
            "Benefit_head_descr": {"type": "string", "description": "Benefit head description"},
            "Country_code(Treatment Country)": {"type": "string", "description": "Treatment country code"}
        }
        
    def init_database(self):
        """Initialize SQLite database for user mappings"""
        conn = sqlite3.connect('user_mappings.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS field_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                data_hash TEXT NOT NULL,
                mappings TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_field_mapping(self, user_id, data_columns, mappings):
        """Save field mapping for a user"""
        data_hash = hashlib.md5(str(sorted(data_columns)).encode()).hexdigest()
        conn = sqlite3.connect('user_mappings.db')
        cursor = conn.cursor()
        
        # Delete existing mapping for this user and data structure
        cursor.execute('DELETE FROM field_mappings WHERE user_id = ? AND data_hash = ?', 
                      (user_id, data_hash))
        
        # Insert new mapping
        cursor.execute('''
            INSERT INTO field_mappings (user_id, data_hash, mappings)
            VALUES (?, ?, ?)
        ''', (user_id, data_hash, json.dumps(mappings)))
        
        conn.commit()
        conn.close()
    
    def load_field_mapping(self, user_id, data_columns):
        """Load field mapping for a user"""
        data_hash = hashlib.md5(str(sorted(data_columns)).encode()).hexdigest()
        conn = sqlite3.connect('user_mappings.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT mappings FROM field_mappings 
            WHERE user_id = ? AND data_hash = ?
            ORDER BY created_at DESC LIMIT 1
        ''', (user_id, data_hash))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def get_user_mapping_count(self, user_id):
        """Get count of saved mappings for a user"""
        conn = sqlite3.connect('user_mappings.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM field_mappings WHERE user_id = ?', (user_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def connect_database(self, db_config):
        """Connect to external database"""
        try:
            if db_config['db_type'] == 'postgresql':
                conn = psycopg2.connect(
                    host=db_config['host'],
                    port=db_config['port'],
                    database=db_config['database'],
                    user=db_config['username'],
                    password=db_config['password']
                )
            elif db_config['db_type'] == 'mysql':
                conn = pymysql.connect(
                    host=db_config['host'],
                    port=int(db_config['port']),
                    database=db_config['database'],
                    user=db_config['username'],
                    password=db_config['password']
                )
            elif db_config['db_type'] == 'sqlserver':
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={db_config['host']},{db_config['port']};DATABASE={db_config['database']};UID={db_config['username']};PWD={db_config['password']}"
                conn = pyodbc.connect(conn_str)
            else:
                raise ValueError("Unsupported database type")
            
            # Test connection and get data
            if db_config.get('custom_query'):
                df = pd.read_sql(db_config['custom_query'], conn, limit=1000)
            elif db_config.get('table_name'):
                df = pd.read_sql(f"SELECT * FROM {db_config['table_name']} LIMIT 1000", conn)
            else:
                # Get first table
                if db_config['db_type'] == 'postgresql':
                    tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' LIMIT 1"
                elif db_config['db_type'] == 'mysql':
                    tables_query = "SHOW TABLES LIMIT 1"
                else:
                    tables_query = "SELECT TOP 1 TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"
                
                table_df = pd.read_sql(tables_query, conn)
                if not table_df.empty:
                    table_name = table_df.iloc[0, 0]
                    df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 1000", conn)
                else:
                    raise ValueError("No tables found in database")
            
            conn.close()
            return df
            
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            return None
    
    def calculate_similarity(self, str1, str2):
        """Calculate similarity between two strings"""
        str1, str2 = str1.lower(), str2.lower()
        
        if str1 == str2:
            return 100
        if str1 in str2 or str2 in str1:
            return 90
        
        words1 = set(str1.replace('_', ' ').replace('-', ' ').split())
        words2 = set(str2.replace('_', ' ').replace('-', ' ').split())
        
        if words1 & words2:
            return int((len(words1 & words2) / len(words1 | words2)) * 80)
        
        return 0
    
    def generate_field_mappings(self, user_id, data_columns):
        """Generate field mappings with user-specific suggestions"""
        # Try to load saved mappings first
        saved_mappings = self.load_field_mapping(user_id, data_columns)
        
        mappings = []
        for required_field, field_info in self.required_fields.items():
            best_match = None
            best_score = 0
            is_confirmed = False
            
            # Check saved mappings first
            if saved_mappings and required_field in saved_mappings:
                saved_field = saved_mappings[required_field]['user_field']
                if saved_field in data_columns:
                    best_match = saved_field
                    best_score = 100
                    is_confirmed = True
            
            # If no saved mapping, find best match
            if not best_match:
                for data_col in data_columns:
                    score = self.calculate_similarity(required_field, data_col)
                    if score > best_score:
                        best_score = score
                        best_match = data_col
                        is_confirmed = score >= 70
            
            mappings.append({
                'required_field': required_field,
                'user_field': best_match or data_columns[0],
                'data_type': field_info['type'],
                'confidence_score': best_score,
                'is_confirmed': is_confirmed,
                'description': field_info['description'],
                'from_saved': saved_mappings is not None and required_field in saved_mappings if saved_mappings else False
            })
        
        return mappings
    
    def create_powerbi_dashboard(self, df, selected_fields, date_range, filters):
        """Create Power BI style dashboard"""
        
        # Apply filters
        filtered_df = df.copy()
        
        # Date filtering
        if date_range and 'Treatment from date' in df.columns:
            try:
                df['Treatment from date'] = pd.to_datetime(df['Treatment from date'], errors='coerce')
                filtered_df = filtered_df[
                    (filtered_df['Treatment from date'] >= date_range[0]) & 
                    (filtered_df['Treatment from date'] <= date_range[1])
                ]
            except:
                pass
        
        # Category filters
        for filter_field, filter_values in filters.items():
            if filter_field in filtered_df.columns and filter_values:
                filtered_df = filtered_df[filtered_df[filter_field].isin(filter_values)]
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Claims</h3>
                <h2>{:,}</h2>
            </div>
            """.format(len(filtered_df)), unsafe_allow_html=True)
        
        with col2:
            total_paid = filtered_df['Paid amount'].sum() if 'Paid amount' in filtered_df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3>Total Paid</h3>
                <h2>${:,.0f}</h2>
            </div>
            """.format(total_paid), unsafe_allow_html=True)
        
        with col3:
            unique_providers = filtered_df['Provider ID'].nunique() if 'Provider ID' in filtered_df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3>Unique Providers</h3>
                <h2>{:,}</h2>
            </div>
            """.format(unique_providers), unsafe_allow_html=True)
        
        with col4:
            avg_claim = filtered_df['Paid amount'].mean() if 'Paid amount' in filtered_df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Claim Amount</h3>
                <h2>${:,.0f}</h2>
            </div>
            """.format(avg_claim), unsafe_allow_html=True)
        
        # Charts Grid
        if selected_fields:
            # Create dynamic charts based on selected fields
            chart_cols = st.columns(2)
            
            for i, field in enumerate(selected_fields[:6]):  # Limit to 6 charts
                with chart_cols[i % 2]:
                    if field in filtered_df.columns:
                        self.create_field_chart(filtered_df, field, f"chart_{i}")
        
        # Combined Analysis
        if len(selected_fields) >= 2:
            st.subheader("🔗 Combined Field Analysis")
            self.create_combined_analysis(filtered_df, selected_fields)
        
        # Business Insights
        st.subheader("💡 Business Insights")
        self.generate_business_insights(filtered_df, selected_fields)
    
    def create_field_chart(self, df, field, chart_key):
        """Create appropriate chart for field"""
        st.subheader(f"📊 {field}")
        
        if df[field].dtype in ['object', 'category']:
            # Categorical field - Bar chart
            value_counts = df[field].value_counts().head(10)
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {field}",
                color=value_counts.values,
                color_continuous_scale="viridis"
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title=field,
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"bar_{chart_key}")
            
            # Insights
            st.info(f"**Top Category:** {value_counts.index[0]} ({value_counts.iloc[0]:,} claims)")
            
        else:
            # Numeric field - Histogram and box plot
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    df, x=field,
                    title=f"Distribution of {field}",
                    nbins=30,
                    color_discrete_sequence=["#667eea"]
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{chart_key}")
            
            with col2:
                fig_box = px.box(
                    df, y=field,
                    title=f"Box Plot of {field}",
                    color_discrete_sequence=["#764ba2"]
                )
                fig_box.update_layout(height=300)
                st.plotly_chart(fig_box, use_container_width=True, key=f"box_{chart_key}")
            
            # Statistics
            stats = df[field].describe()
            st.info(f"**Mean:** {stats['mean']:.2f} | **Median:** {stats['50%']:.2f} | **Std:** {stats['std']:.2f}")
    
    def create_combined_analysis(self, df, selected_fields):
        """Create combined analysis charts"""
        numeric_fields = [f for f in selected_fields if f in df.columns and df[f].dtype in ['int64', 'float64']]
        categorical_fields = [f for f in selected_fields if f in df.columns and df[f].dtype == 'object']
        
        if len(numeric_fields) >= 2:
            # Correlation heatmap
            corr_data = df[numeric_fields].corr()
            fig_corr = px.imshow(
                corr_data,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        if len(numeric_fields) >= 2:
            # Scatter plot
            fig_scatter = px.scatter(
                df, x=numeric_fields[0], y=numeric_fields[1],
                title=f"{numeric_fields[0]} vs {numeric_fields[1]}",
                color=categorical_fields[0] if categorical_fields else None,
                size=numeric_fields[2] if len(numeric_fields) > 2 else None
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def generate_business_insights(self, df, selected_fields):
        """Generate business insights"""
        insights = []
        
        # Volume insights
        if 'Treatment from date' in df.columns:
            try:
                df['Treatment from date'] = pd.to_datetime(df['Treatment from date'], errors='coerce')
                monthly_claims = df.groupby(df['Treatment from date'].dt.to_period('M')).size()
                if len(monthly_claims) > 1:
                    trend = "increasing" if monthly_claims.iloc[-1] > monthly_claims.iloc[0] else "decreasing"
                    insights.append(f"📈 Claims volume is {trend} over time")
            except:
                pass
        
        # Provider concentration
        if 'Provider ID' in df.columns:
            provider_claims = df['Provider ID'].value_counts()
            top_provider_pct = (provider_claims.iloc[0] / len(df)) * 100
            if top_provider_pct > 20:
                insights.append(f"⚠️ High provider concentration: Top provider handles {top_provider_pct:.1f}% of claims")
        
        # Amount insights
        if 'Paid amount' in df.columns:
            high_value_claims = (df['Paid amount'] > df['Paid amount'].quantile(0.95)).sum()
            insights.append(f"💰 {high_value_claims} high-value claims (top 5%) identified")
        
        # Display insights
        for insight in insights:
            st.success(insight)
    
    def create_forecasting_analysis(self, df):
        """Create forecasting analysis with time series models"""
        st.subheader("🔮 Predictive Analytics & Forecasting")
        
        # Date column selection
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if not date_columns:
            st.warning("No date columns found for forecasting analysis.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Select Date Column", date_columns)
        with col2:
            metric_col = st.selectbox("Select Metric to Forecast", 
                                    [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
        
        if date_col and metric_col:
            try:
                # Prepare time series data
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                ts_data = df.groupby(df[date_col].dt.date)[metric_col].agg(['sum', 'count', 'mean']).reset_index()
                ts_data.columns = ['date', 'total', 'count', 'average']
                ts_data = ts_data.sort_values('date')
                
                if len(ts_data) < 10:
                    st.warning("Insufficient data points for reliable forecasting (minimum 10 required).")
                    return
                
                # Forecasting parameters
                col1, col2, col3 = st.columns(3)
                with col1:
                    forecast_metric = st.selectbox("Forecast Metric", ['total', 'count', 'average'])
                with col2:
                    forecast_periods = st.slider("Forecast Periods", 5, 30, 10)
                with col3:
                    model_type = st.selectbox("Model Type", ['Exponential Smoothing', 'Linear Trend', 'ARIMA'])
                
                # Prepare data
                y = ts_data[forecast_metric].values
                dates = pd.to_datetime(ts_data['date'])
                
                # Split data for validation
                train_size = int(len(y) * 0.8)
                y_train, y_test = y[:train_size], y[train_size:]
                dates_train, dates_test = dates[:train_size], dates[train_size:]
                
                # Generate forecasts
                forecasts = {}
                
                if model_type == 'Exponential Smoothing':
                    try:
                        model = ETSModel(y_train, trend='add', seasonal=None)
                        fitted_model = model.fit()
                        forecast = fitted_model.forecast(len(y_test) + forecast_periods)
                        forecasts['Exponential Smoothing'] = forecast
                    except:
                        st.warning("Exponential Smoothing failed, using Linear Trend")
                        model_type = 'Linear Trend'
                
                if model_type == 'Linear Trend':
                    X_train = np.arange(len(y_train)).reshape(-1, 1)
                    X_forecast = np.arange(len(y_train), len(y_train) + len(y_test) + forecast_periods).reshape(-1, 1)
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    forecast = model.predict(X_forecast)
                    forecasts['Linear Trend'] = forecast
                
                elif model_type == 'ARIMA':
                    try:
                        model = ARIMA(y_train, order=(1, 1, 1))
                        fitted_model = model.fit()
                        forecast = fitted_model.forecast(len(y_test) + forecast_periods)
                        forecasts['ARIMA'] = forecast
                    except:
                        st.warning("ARIMA failed, using Linear Trend")
                        X_train = np.arange(len(y_train)).reshape(-1, 1)
                        X_forecast = np.arange(len(y_train), len(y_train) + len(y_test) + forecast_periods).reshape(-1, 1)
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        forecast = model.predict(X_forecast)
                        forecasts['Linear Trend'] = forecast
                
                # Create forecast visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=dates_train, y=y_train,
                    mode='lines+markers',
                    name='Training Data',
                    line=dict(color='blue')
                ))
                
                if len(y_test) > 0:
                    fig.add_trace(go.Scatter(
                        x=dates_test, y=y_test,
                        mode='lines+markers',
                        name='Actual Test Data',
                        line=dict(color='green')
                    ))
                
                # Forecast
                future_dates = pd.date_range(start=dates.iloc[-1], periods=forecast_periods+1, freq='D')[1:]
                all_forecast_dates = list(dates_test) + list(future_dates)
                
                for model_name, forecast_values in forecasts.items():
                    fig.add_trace(go.Scatter(
                        x=all_forecast_dates, y=forecast_values,
                        mode='lines+markers',
                        name=f'{model_name} Forecast',
                        line=dict(dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"Forecasting Analysis - {forecast_metric.title()} {metric_col}",
                    xaxis_title="Date",
                    yaxis_title=forecast_metric.title(),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model Performance
                if len(y_test) > 0:
                    st.subheader("📊 Model Performance")
                    
                    for model_name, forecast_values in forecasts.items():
                        test_forecast = forecast_values[:len(y_test)]
                        mae = mean_absolute_error(y_test, test_forecast)
                        rmse = np.sqrt(mean_squared_error(y_test, test_forecast))
                        mape = np.mean(np.abs((y_test - test_forecast) / y_test)) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{model_name} MAE", f"{mae:.2f}")
                        with col2:
                            st.metric(f"{model_name} RMSE", f"{rmse:.2f}")
                        with col3:
                            st.metric(f"{model_name} MAPE", f"{mape:.1f}%")
                
                # Business Insights
                st.subheader("💡 Forecasting Insights")
                
                current_avg = np.mean(y_train[-7:]) if len(y_train) >= 7 else np.mean(y_train)
                forecast_avg = np.mean(list(forecasts.values())[0][-forecast_periods:])
                
                change_pct = ((forecast_avg - current_avg) / current_avg) * 100
                
                if abs(change_pct) > 10:
                    trend_direction = "increase" if change_pct > 0 else "decrease"
                    st.warning(f"📈 Significant {trend_direction} expected: {abs(change_pct):.1f}% change in {forecast_metric}")
                else:
                    st.info(f"📊 Stable trend expected: {abs(change_pct):.1f}% change in {forecast_metric}")
                
                # Peak prediction
                forecast_values = list(forecasts.values())[0][-forecast_periods:]
                peak_idx = np.argmax(forecast_values)
                peak_date = future_dates[peak_idx]
                st.success(f"🎯 Peak {forecast_metric} expected on: {peak_date.strftime('%Y-%m-%d')}")
                
            except Exception as e:
                st.error(f"Forecasting analysis failed: {str(e)}")

def main():
    load_custom_css()
    
    # Initialize analytics
    analytics = HealthcareAnalytics()
    
    # Sidebar for user selection and controls
    with st.sidebar:
        st.title("🏥 Healthcare Analytics")
        
        # User Selection
        st.subheader("👤 User Profile")
        selected_user = st.selectbox(
            "Select User Profile",
            ["User 1", "User 2", "User 3"],
            help="Select your user profile to save and load field mappings"
        )
        
        # Show user mapping count
        mapping_count = analytics.get_user_mapping_count(selected_user)
        st.info(f"📊 {mapping_count} saved mappings for {selected_user}")
        
        st.divider()
        
        # Data Upload Section
        st.subheader("📁 Data Upload")
        
        upload_method = st.radio("Choose data source:", ["Upload File", "Connect Database"])
        
        uploaded_data = None
        
        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload your healthcare claims data"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        uploaded_data = pd.read_csv(uploaded_file)
                    else:
                        uploaded_data = pd.read_excel(uploaded_file)
                    st.success(f"✅ File uploaded: {len(uploaded_data)} rows")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        else:  # Database connection
            with st.expander("🔗 Database Configuration"):
                db_type = st.selectbox("Database Type", 
                                     ["postgresql", "mysql", "sqlserver", "sqlite"])
                host = st.text_input("Host", "localhost")
                port = st.text_input("Port", "5432" if db_type == "postgresql" else "3306")
                database = st.text_input("Database Name")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                col1, col2 = st.columns(2)
                with col1:
                    table_name = st.text_input("Table Name (optional)")
                with col2:
                    custom_query = st.text_area("Custom SQL Query (optional)")
                
                if st.button("🔗 Connect & Load Data"):
                    db_config = {
                        'db_type': db_type,
                        'host': host,
                        'port': port,
                        'database': database,
                        'username': username,
                        'password': password,
                        'table_name': table_name,
                        'custom_query': custom_query
                    }
                    
                    uploaded_data = analytics.connect_database(db_config)
                    if uploaded_data is not None:
                        st.success(f"✅ Database connected: {len(uploaded_data)} rows")
    
    # Main content area
    if uploaded_data is not None:
        # Store data in session state
        st.session_state.uploaded_data = uploaded_data
        st.session_state.selected_user = selected_user
        
        # Navigation tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ Field Mapping", 
            "📊 Claims Data Summary", 
            "🔍 FWA Detection", 
            "🔮 Predictive Analytics"
        ])
        
        with tab1:
            st.header("🗺️ Field Mapping")
            
            # Generate field mappings
            if 'field_mappings' not in st.session_state:
                st.session_state.field_mappings = analytics.generate_field_mappings(
                    selected_user, uploaded_data.columns.tolist()
                )
            
            # Pagination for field mappings
            fields_per_page = 15
            total_fields = len(st.session_state.field_mappings)
            total_pages = (total_fields - 1) // fields_per_page + 1
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                current_page = st.selectbox(
                    f"Page (showing {fields_per_page} fields per page)",
                    range(1, total_pages + 1),
                    format_func=lambda x: f"Page {x} of {total_pages}"
                )
            
            # Display current page fields
            start_idx = (current_page - 1) * fields_per_page
            end_idx = min(start_idx + fields_per_page, total_fields)
            current_mappings = st.session_state.field_mappings[start_idx:end_idx]
            
            # Mapping interface
            st.subheader(f"Field Mapping - Page {current_page}")
            
            # Header
            col1, col2, col3, col4, col5 = st.columns([3, 3, 2, 2, 1])
            with col1:
                st.write("**Required Field**")
            with col2:
                st.write("**Your Data Field**")
            with col3:
                st.write("**Confidence**")
            with col4:
                st.write("**Source**")
            with col5:
                st.write("**Confirm**")
            
            # Mapping rows
            for i, mapping in enumerate(current_mappings):
                actual_idx = start_idx + i
                col1, col2, col3, col4, col5 = st.columns([3, 3, 2, 2, 1])
                
                with col1:
                    st.write(mapping['required_field'])
                    st.caption(mapping['description'])
                
                with col2:
                    new_field = st.selectbox(
                        "Field",
                        uploaded_data.columns.tolist(),
                        index=uploaded_data.columns.tolist().index(mapping['user_field']),
                        key=f"field_{actual_idx}",
                        label_visibility="collapsed"
                    )
                    st.session_state.field_mappings[actual_idx]['user_field'] = new_field
                
                with col3:
                    confidence = mapping['confidence_score']
                    if confidence >= 70:
                        st.success(f"{confidence}%")
                    elif confidence >= 40:
                        st.warning(f"{confidence}%")
                    else:
                        st.error(f"{confidence}%")
                
                with col4:
                    if mapping.get('from_saved', False):
                        st.info("💾 Saved")
                    else:
                        st.write("🔍 Auto")
                
                with col5:
                    confirmed = st.checkbox(
                        "Confirm",
                        value=mapping['is_confirmed'],
                        key=f"confirm_{actual_idx}",
                        label_visibility="collapsed"
                    )
                    st.session_state.field_mappings[actual_idx]['is_confirmed'] = confirmed
            
            # Action buttons
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Mapping Only", type="secondary"):
                    # Save current mappings
                    mapping_dict = {
                        mapping['required_field']: {
                            'user_field': mapping['user_field'],
                            'is_confirmed': mapping['is_confirmed']
                        }
                        for mapping in st.session_state.field_mappings
                    }
                    analytics.save_field_mapping(selected_user, uploaded_data.columns.tolist(), mapping_dict)
                    st.success("✅ Field mapping saved successfully!")
            
            with col2:
                if st.button("✅ Confirm All", type="secondary"):
                    for mapping in st.session_state.field_mappings:
                        mapping['is_confirmed'] = True
                    st.rerun()
            
            with col3:
                if st.button("🚀 Confirm Mapping", type="primary"):
                    # Save mappings and proceed
                    mapping_dict = {
                        mapping['required_field']: {
                            'user_field': mapping['user_field'],
                            'is_confirmed': mapping['is_confirmed']
                        }
                        for mapping in st.session_state.field_mappings
                    }
                    analytics.save_field_mapping(selected_user, uploaded_data.columns.tolist(), mapping_dict)
                    st.session_state.mapping_confirmed = True
                    st.success("✅ Field mapping confirmed and saved!")
        
        with tab2:
            if st.session_state.get('mapping_confirmed', False):
                st.header("📊 Claims Data Summary Dashboard")
                
                # Dashboard controls in sidebar
                with st.sidebar:
                    st.divider()
                    st.subheader("🎛️ Dashboard Controls")
                    
                    # Field selection
                    available_fields = [col for col in uploaded_data.columns if uploaded_data[col].notna().any()]
                    selected_fields = st.multiselect(
                        "Select Fields for Analysis",
                        available_fields,
                        default=available_fields[:3] if len(available_fields) >= 3 else available_fields,
                        help="Choose fields to analyze in the dashboard"
                    )
                    
                    # Date range filter
                    date_columns = [col for col in uploaded_data.columns if 'date' in col.lower()]
                    if date_columns:
                        date_col = st.selectbox("Date Column for Filtering", date_columns)
                        if date_col:
                            try:
                                uploaded_data[date_col] = pd.to_datetime(uploaded_data[date_col], errors='coerce')
                                min_date = uploaded_data[date_col].min().date()
                                max_date = uploaded_data[date_col].max().date()
                                date_range = st.date_input(
                                    "Date Range",
                                    value=(min_date, max_date),
                                    min_value=min_date,
                                    max_value=max_date
                                )
                            except:
                                date_range = None
                    else:
                        date_range = None
                    
                    # Category filters
                    categorical_fields = [col for col in uploaded_data.columns if uploaded_data[col].dtype == 'object'][:3]
                    filters = {}
                    for field in categorical_fields:
                        unique_values = uploaded_data[field].dropna().unique()[:20]  # Limit options
                        selected_values = st.multiselect(
                            f"Filter by {field}",
                            unique_values,
                            help=f"Filter data by {field} values"
                        )
                        if selected_values:
                            filters[field] = selected_values
                
                # Create dashboard
                analytics.create_powerbi_dashboard(uploaded_data, selected_fields, date_range, filters)
                
            else:
                st.warning("⚠️ Please complete field mapping first to access the dashboard.")
        
        with tab3:
            if st.session_state.get('mapping_confirmed', False):
                st.header("🔍 FWA Detection Analysis")
                
                # Initialize FWA scenarios
                fwa_scenarios = FWAScenarios()
                
                # Scenario selection
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🐍 Python Rule-Based Scenarios")
                    python_scenarios = list(fwa_scenarios.python_scenarios.keys())
                    
                    # Control buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✅ Select All Python"):
                            st.session_state.selected_python = python_scenarios
                    with col_b:
                        if st.button("❌ Clear All Python"):
                            st.session_state.selected_python = []
                    
                    # Scenario selection
                    selected_python = st.multiselect(
                        "Choose Python Scenarios",
                        python_scenarios,
                        default=st.session_state.get('selected_python', []),
                        key="python_scenarios_select"
                    )
                    st.session_state.selected_python = selected_python
                
                with col2:
                    st.subheader("🤖 ML-Based Scenarios")
                    ml_scenarios = list(fwa_scenarios.ml_scenarios.keys())
                    
                    # Control buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✅ Select All ML"):
                            st.session_state.selected_ml = ml_scenarios
                    with col_b:
                        if st.button("❌ Clear All ML"):
                            st.session_state.selected_ml = []
                    
                    # Scenario selection
                    selected_ml = st.multiselect(
                        "Choose ML Scenarios",
                        ml_scenarios,
                        default=st.session_state.get('selected_ml', []),
                        key="ml_scenarios_select"
                    )
                    st.session_state.selected_ml = selected_ml
                
                # Run analysis
                if st.button("🚀 Run FWA Analysis", type="primary"):
                    if not selected_python and not selected_ml:
                        st.warning("Please select at least one scenario to run.")
                    else:
                        with st.spinner("Running FWA analysis..."):
                            try:
                                # Create field mapping dictionary
                                field_mapping = {
                                    mapping['required_field']: mapping['user_field']
                                    for mapping in st.session_state.field_mappings
                                    if mapping['is_confirmed']
                                }
                                
                                # Run scenarios
                                results = fwa_scenarios.run_scenarios(
                                    uploaded_data, 
                                    selected_python, 
                                    selected_ml, 
                                    field_mapping
                                )
                                
                                if results:
                                    st.success("✅ FWA Analysis completed successfully!")
                                    
                                    # Display results summary
                                    total_flagged = len(results)
                                    st.metric("🚨 Total Flagged Claims", total_flagged)
                                    
                                    # Create downloadable Excel file
                                    excel_buffer = fwa_scenarios.create_excel_report(results)
                                    
                                    st.download_button(
                                        label="📥 Download FWA Analysis Report",
                                        data=excel_buffer,
                                        file_name=f"FWA_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                    
                                    # Show sample results
                                    st.subheader("📋 Sample Flagged Claims")
                                    st.dataframe(results.head(10))
                                    
                                else:
                                    st.info("ℹ️ No fraudulent claims detected with current scenarios.")
                                
                            except Exception as e:
                                st.error(f"❌ FWA Analysis failed: {str(e)}")
                                st.error("All scenarios stopped due to error.")
                
                # FWA Prevention Recommendations
                st.divider()
                if st.button("💡 Generate FWA Prevention Recommendations"):
                    st.subheader("🛡️ FWA Prevention Recommendations")
                    
                    recommendations = [
                        "**Enhanced Monitoring**: Implement real-time monitoring for high-risk providers and unusual claim patterns",
                        "**Automated Screening**: Deploy ML-based screening for all incoming claims before processing",
                        "**Provider Education**: Conduct regular training sessions on proper billing practices and compliance",
                        "**Regular Audits**: Schedule quarterly audits for providers with high claim volumes or amounts",
                        "**Strengthened Controls**: Implement multi-level approval processes for high-value claims",
                        "**Data Analytics**: Use predictive analytics to identify emerging fraud patterns",
                        "**Cross-referencing**: Implement cross-referencing with external databases for provider verification",
                        "**Whistleblower Program**: Establish anonymous reporting mechanisms for suspected fraud"
                    ]
                    
                    for rec in recommendations:
                        st.success(rec)
            else:
                st.warning("⚠️ Please complete field mapping first to access FWA detection.")
        
        with tab4:
            if st.session_state.get('mapping_confirmed', False):
                analytics.create_forecasting_analysis(uploaded_data)
            else:
                st.warning("⚠️ Please complete field mapping first to access predictive analytics.")
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h1 style="color: white; font-size: 3em; margin-bottom: 20px;">🏥 Healthcare Analytics Tool</h1>
            <p style="color: white; font-size: 1.2em; margin-bottom: 30px;">
                Advanced FWA Detection & Predictive Analytics Platform
            </p>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 30px; border-radius: 15px; backdrop-filter: blur(10px);">
                <h3 style="color: white;">🚀 Get Started</h3>
                <p style="color: white;">Upload your healthcare claims data or connect to your database to begin analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()