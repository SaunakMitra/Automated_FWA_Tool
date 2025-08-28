# Comprehensive Healthcare Analytics Tool - Python Implementation

This is a complete Python implementation of the Healthcare Analytics Tool using Streamlit with advanced FWA detection capabilities.

## 🚀 Features

### Core Functionality
- **Interactive Dashboard**: Clean interface with network-inspired background styling
- **File Upload**: Support for CSV and Excel files
- **Database Connection**: Connect to PostgreSQL, MySQL, SQL Server, and Oracle databases
- **Data Preview**: Shows first 5 rows and comprehensive data summary
- **Advanced Field Mapping**: Intelligent mapping with confidence scoring (0-100%) and color coding

### Field Mapping
Supports comprehensive healthcare data fields including:
- **Claim Information**: Claim ID, Member ID, Provider ID, Invoice details
- **Medical Codes**: ICD-10 Diagnostic codes, CPT Procedure codes, NDC codes
- **Financial Data**: Paid amounts, Billed amounts, Deductibles, Copays
- **Temporal Data**: Service dates, Admission/Discharge dates, Policy periods
- **Geographic Data**: Member and Provider zip codes
- **Policy Information**: Coverage types, Network types, Policy details
- **Authorization Data**: Prior auth flags, Referral information

### FWA Detection Scenarios

#### Python-Based Rules (Hypothesis Sheet)
1. **Duplicate Claims Detection**: Identifies potential duplicate submissions
2. **Billing Pattern Analysis**: Analyzes unusual billing patterns and frequency
3. **Age-Service Mismatch**: Detects services inappropriate for patient age
4. **Provider Network Analysis**: Identifies suspicious provider-patient relationships
5. **Amount Outlier Detection**: Detects claims with unusually high amounts
6. **Frequency Analysis**: Identifies providers with unusually high claim frequency
7. **Geographic Anomaly Detection**: Detects claims from geographically distant providers
8. **Prior Authorization Violations**: Identifies claims without required prior authorization

#### ML-Based Scenarios (ML Scenarios Sheet)
1. **Anomaly Detection (Isolation Forest)**: Uses Isolation Forest to detect anomalous claims
2. **Clustering Analysis (DBSCAN)**: Identifies unusual claim clusters
3. **Fraud Prediction Model**: Predicts fraud probability using Random Forest
4. **Pattern Recognition**: Identifies suspicious patterns in provider behavior

### Advanced Analytics
- **Fraud Scoring**: Calculates overall fraud score (0-100%) based on flagged scenarios
- **Claims Ranking**: Sorts claims by fraud score (highest risk first)
- **Excel Export**: Downloads results in two sheets (Python Rules & ML Scenarios)
- **Binary Flagging**: Uses 0/1 flags for each scenario
- **Visualization**: Interactive charts and graphs for data analysis

### Dashboard Features
- **Claims Data Summary**: Complete statistical analysis with visualizations
- **Trend Analysis**: Time-series analysis of claim patterns
- **Real-time Recommendations**: Suggests actions to prevent FWA

## 🎨 Design Features

### Background Styling
- Network-inspired gradient background (blue to orange)
- Glass-morphism effect on containers
- Professional healthcare color scheme
- Responsive design with smooth animations

### Confidence Scoring
- **Green (91-100%)**: High confidence mapping
- **Yellow (41-90%)**: Medium confidence mapping
- **Red (0-40%)**: Low confidence mapping

## 📊 Output Format

### Excel File Structure
The tool generates an Excel file with two sheets:

#### 1. Python_Rules Sheet
- Claim ID, Member ID, Provider ID, Paid Amount, Fraud Score
- Binary flags (0/1) for each Python-based scenario
- Sorted by fraud score (descending)

#### 2. ML_Scenarios Sheet
- Claim ID, Member ID, Provider ID, Paid Amount, Fraud Score
- Binary flags (0/1) for each ML-based scenario
- Fraud probability scores for applicable algorithms

## 🛠️ Installation

1. **Install Python 3.8+**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   streamlit run healthcare_analytics.py
   ```

## 📈 Usage Workflow

1. **Upload Data**: Use file upload or database connection
2. **Data Preview**: Review data structure and summary statistics
3. **Field Mapping**: Map your data fields to required healthcare fields
4. **Select Scenarios**: Choose from available Python rules and ML algorithms
5. **Run Analysis**: Execute selected FWA detection scenarios
6. **Download Results**: Get Excel file with flagged claims and fraud scores
7. **Review Recommendations**: Follow suggested actions to prevent FWA

## 🔍 FWA Detection Logic

### Python Rules
- **Statistical Analysis**: Uses IQR, standard deviation, and percentile-based detection
- **Pattern Matching**: Identifies duplicate patterns and unusual frequencies
- **Business Rules**: Applies healthcare-specific validation rules
- **Geographic Analysis**: Detects distance-based anomalies

### ML Algorithms
- **Isolation Forest**: Unsupervised anomaly detection
- **DBSCAN Clustering**: Density-based clustering for outlier detection
- **Random Forest**: Supervised learning for fraud prediction
- **Feature Engineering**: Automated encoding of categorical variables

## 📋 Recommendations Engine

The tool provides actionable recommendations based on analysis results:
- Enhanced monitoring for high-risk claims
- Automated screening implementation
- Provider education programs
- Regular audit schedules
- Strengthened control measures

## 🔧 Customization

### Adding New Scenarios
1. Add scenario definition to `python_scenarios` or `ml_scenarios`
2. Implement detection logic in respective run methods
3. Update field requirements and descriptions

### Modifying Field Mappings
1. Update `required_fields` dictionary
2. Adjust similarity calculation logic
3. Customize confidence scoring thresholds

### Styling Customization
1. Modify CSS in `load_custom_css()` function
2. Update color schemes and gradients
3. Adjust card layouts and animations

This comprehensive tool provides healthcare organizations with powerful FWA detection capabilities while maintaining an intuitive, professional interface.