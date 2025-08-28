# Healthcare Analytics Tool - Python Implementation

This is a complete Python implementation of the Healthcare Analytics Tool using Streamlit.

## Features

- **Interactive Dashboard**: Clean interface with upload options and three main analytical modules
- **File Upload**: Support for CSV and Excel files
- **Database Connection**: Connect to PostgreSQL, MySQL, SQL Server, and Oracle databases
- **Data Preview**: Shows first 5 rows and comprehensive data summary
- **Field Mapping**: Intelligent field mapping with confidence scoring and color coding
- **Claims Data Summary**: Complete statistical analysis similar to pandas describe()
- **FWA Detection**: Fraud, Waste, and Abuse detection with multiple scenarios
- **Trend Analysis**: Time-series analysis capabilities

## Installation

1. Install Python 3.8 or higher
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run healthcare_analytics.py
```

## Code Structure

### Main Components

1. **HealthcareAnalytics Class**: Core logic for data processing and analysis
2. **Data Loading**: File upload and database connection functionality
3. **Field Mapping**: Intelligent mapping with similarity scoring
4. **Analytics Modules**: 
   - Claims Data Summary with visualizations
   - FWA Detection with scenario selection
   - Trend Analysis capabilities

### Key Methods

- `load_file_data()`: Handles CSV/Excel file uploads
- `connect_database()`: Database connection with SQLAlchemy
- `generate_data_summary()`: Creates comprehensive data statistics
- `generate_field_mappings()`: Intelligent field mapping with confidence scores
- `get_available_scenarios()`: Determines which FWA scenarios can run
- `create_visualization()`: Generates appropriate charts for each data type

### Confidence Scoring

- **Red (0-40%)**: Low confidence mapping
- **Yellow (41-90%)**: Medium confidence mapping  
- **Green (91-100%)**: High confidence mapping

### FWA Scenarios

1. **Duplicate Claims Detection**: Identifies potential duplicate submissions
2. **Billing Pattern Analysis**: Analyzes unusual billing patterns
3. **Age-Service Mismatch**: Detects inappropriate services for patient age
4. **Provider Network Analysis**: Identifies suspicious relationships
5. **Amount Outlier Detection**: Detects unusually high claim amounts

## Database Support

Supports major databases:
- PostgreSQL
- MySQL
- SQL Server
- Oracle

## Visualizations

- Histograms for numerical data
- Bar charts for categorical data
- Interactive Plotly charts
- Statistical summaries with pandas

## Session State Management

Uses Streamlit's session state to maintain:
- Current application step
- Data loading status
- Field mapping confirmation
- User selections and preferences

This implementation provides all the functionality you requested with a clean, professional interface suitable for healthcare analytics.