import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class PythonScenarios:
    """Python-based FWA detection scenarios"""
    
    def __init__(self):
        self.scenarios = {
            "Scenario 1 - Amount Outlier Detection": {
                "description": "Detects claims with unusually high amounts using IQR analysis",
                "required_fields": ["Claimed_currency_code", "Provider_country_code", "Payee type", "Claim invoice gross total amount", "Procedure_code (CPT codes)", "Incident count"],
                "function": self.scenario_1_amount_outlier
            },
            "Scenario 2 - Chemotherapy Frequency Analysis": {
                "description": "Identifies suspicious chemotherapy treatment patterns",
                "required_fields": ["Procedure_code (CPT codes)", "Treatment from date", "Treatment to date", "Member ID", "Claim ID"],
                "function": self.scenario_2_chemotherapy
            },
            "Scenario 3 - Cross-Country Treatment Analysis": {
                "description": "Detects treatments in different countries on same date",
                "required_fields": ["Member ID", "Treatment from date", "Provider_country_code", "Diagnostic_code (ICD-10)"],
                "function": self.scenario_3_cross_country
            },
            "Scenario 4 - Sunday Hospital Claims": {
                "description": "Identifies hospital claims submitted on Sundays",
                "required_fields": ["Treatment from date", "Provider ID", "Claim ID"],
                "function": self.scenario_4_sunday_claims
            },
            "Scenario 5 - Invalid Invoice References": {
                "description": "Detects duplicate claims with invalid invoice references",
                "required_fields": ["Member ID", "Invoice No Reference", "Claim ID", "Provider ID"],
                "function": self.scenario_5_invalid_invoices
            },
            "Scenario 6 - Inpatient-Outpatient Same Day": {
                "description": "Identifies same-day inpatient and outpatient treatments",
                "required_fields": ["Member ID", "Treatment from date", "Treatment to date", "Coverage type (Inpatient, Outpatient, Pharmacy, etc.)"],
                "function": self.scenario_6_same_day_treatments
            },
            "Scenario 7 - Multi-Country Provider Analysis": {
                "description": "Detects providers operating in multiple countries",
                "required_fields": ["Provider ID", "Provider_country_code", "Claim ID"],
                "function": self.scenario_7_multi_country
            },
            "Scenario 8 - Multiple Provider Same Day": {
                "description": "Identifies members visiting multiple providers on same day",
                "required_fields": ["Member ID", "Treatment from date", "Provider ID", "Payment currency code"],
                "function": self.scenario_8_multiple_providers
            }
        }
    
    def get_available_scenarios(self):
        return self.scenarios
    
    def run_scenario(self, scenario_name, df):
        """Run a specific scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario_func = self.scenarios[scenario_name]["function"]
        return scenario_func(df)
    
    def map_column_names(self, df, required_columns):
        """Map column names to handle variations"""
        column_mapping = {}
        df_columns = df.columns.tolist()
        
        for req_col in required_columns:
            # Find best matching column
            best_match = None
            best_score = 0
            
            for df_col in df_columns:
                score = self.calculate_similarity(req_col.lower(), df_col.lower())
                if score > best_score:
                    best_score = score
                    best_match = df_col
            
            if best_score > 0.5:  # Minimum similarity threshold
                column_mapping[req_col] = best_match
            else:
                column_mapping[req_col] = req_col  # Keep original if no good match
        
        return column_mapping
    
    def calculate_similarity(self, str1, str2):
        """Calculate similarity between two strings"""
        str1, str2 = str1.lower(), str2.lower()
        
        if str1 == str2:
            return 1.0
        if str1 in str2 or str2 in str1:
            return 0.9
        
        words1 = set(str1.replace('_', ' ').replace('-', ' ').split())
        words2 = set(str2.replace('_', ' ').replace('-', ' ').split())
        
        if words1 & words2:
            return len(words1 & words2) / len(words1 | words2)
        
        return 0.0
    
    def scenario_1_amount_outlier(self, df):
        """Scenario 1: Amount Outlier Detection"""
        try:
            # Map column names
            required_cols = ["Claimed_currency_code", "Provider_country_code", "Payee type", 
                           "Claim invoice gross total amount", "Procedure_code (CPT codes)", "Incident count"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Check if required columns exist
            missing_cols = [col for col in required_cols if col_mapping[col] not in df.columns]
            if missing_cols:
                return pd.DataFrame()  # Return empty if columns missing
            
            # Filter base dataset
            filtered = df[
                (df[col_mapping["Claimed_currency_code"]].astype(str) == "GBP") &
                (df[col_mapping["Provider_country_code"]].astype(str) == "UK") &
                (df[col_mapping["Payee type"]].astype(str) == "P") &
                (pd.to_numeric(df[col_mapping["Claim invoice gross total amount"]], errors='coerce') < 10000) &
                (df[col_mapping["Procedure_code (CPT codes)"]].notnull())
            ].copy()
            
            if filtered.empty:
                return pd.DataFrame()
            
            # Calculate gross per incident
            filtered['gross_per_incident'] = (
                pd.to_numeric(filtered[col_mapping["Claim invoice gross total amount"]], errors='coerce') / 
                pd.to_numeric(filtered[col_mapping["Incident count"]], errors='coerce').replace(0, np.nan)
            )
            
            # Compute IQR statistics
            procedure_stats = (
                filtered.groupby(col_mapping["Procedure_code (CPT codes)"])['gross_per_incident']
                .agg(['quantile'], q=[0.25, 0.75])
                .reset_index()
            )
            
            if not procedure_stats.empty:
                procedure_stats['iqr_threshold'] = (
                    procedure_stats[('gross_per_incident', 'quantile', 0.75)] + 
                    1.5 * (procedure_stats[('gross_per_incident', 'quantile', 0.75)] - 
                           procedure_stats[('gross_per_incident', 'quantile', 0.25)])
                )
                
                # Flag outliers
                flagged = filtered[filtered['gross_per_incident'] > filtered['gross_per_incident'].quantile(0.95)]
                return flagged[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame()
            
        except Exception as e:
            raise Exception(f"Scenario 1 failed: {str(e)}")
    
    def scenario_2_chemotherapy(self, df):
        """Scenario 2: Chemotherapy Frequency Analysis"""
        try:
            required_cols = ["Procedure_code (CPT codes)", "Treatment from date", "Treatment to date", "Member ID", "Claim ID"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Filter chemotherapy claims
            chemo_claims = df[
                (df[col_mapping["Procedure_code (CPT codes)"]].astype(str) == "4030") &
                (df[col_mapping["Treatment from date"]].notnull()) &
                (df[col_mapping["Treatment to date"]].notnull())
            ].copy()
            
            if chemo_claims.empty:
                return pd.DataFrame()
            
            # Convert dates
            chemo_claims[col_mapping["Treatment from date"]] = pd.to_datetime(
                chemo_claims[col_mapping["Treatment from date"]], errors='coerce'
            )
            
            # Sort and calculate gaps
            chemo_claims = chemo_claims.sort_values([col_mapping["Member ID"], col_mapping["Treatment from date"]])
            chemo_claims['previous_treatment'] = chemo_claims.groupby(col_mapping["Member ID"])[col_mapping["Treatment from date"]].shift(1)
            chemo_claims['days_between'] = (chemo_claims[col_mapping["Treatment from date"]] - chemo_claims['previous_treatment']).dt.days
            
            # Filter suspicious gaps (3-13 days)
            suspicious = chemo_claims[
                (chemo_claims['days_between'] >= 3) & 
                (chemo_claims['days_between'] <= 13)
            ]
            
            return suspicious[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
        except Exception as e:
            raise Exception(f"Scenario 2 failed: {str(e)}")
    
    def scenario_3_cross_country(self, df):
        """Scenario 3: Cross-Country Treatment Analysis"""
        try:
            required_cols = ["Member ID", "Treatment from date", "Provider_country_code", "Diagnostic_code (ICD-10)"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Self-join on Member ID and Treatment date
            df_copy = df.copy()
            merged = df.merge(
                df_copy,
                on=[col_mapping["Member ID"], col_mapping["Treatment from date"]],
                suffixes=('_t1', '_t2')
            )
            
            # Filter different countries and diagnoses
            cross_country = merged[
                (merged[f'{col_mapping["Provider_country_code"]}_t1'] != merged[f'{col_mapping["Provider_country_code"]}_t2']) &
                (merged[f'{col_mapping["Diagnostic_code (ICD-10)"]}_t1'] != merged[f'{col_mapping["Diagnostic_code (ICD-10)"]}_t2'])
            ]
            
            if not cross_country.empty:
                return cross_country[['Claim ID_t1', 'Member ID', 'Provider ID_t1', 'Paid amount_t1']].rename(columns={
                    'Claim ID_t1': 'Claim ID',
                    'Provider ID_t1': 'Provider ID',
                    'Paid amount_t1': 'Paid amount'
                }).drop_duplicates()
            
            return pd.DataFrame()
            
        except Exception as e:
            raise Exception(f"Scenario 3 failed: {str(e)}")
    
    def scenario_4_sunday_claims(self, df):
        """Scenario 4: Sunday Hospital Claims"""
        try:
            required_cols = ["Treatment from date", "Provider ID", "Claim ID"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Convert to datetime
            df[col_mapping["Treatment from date"]] = pd.to_datetime(df[col_mapping["Treatment from date"]], errors='coerce')
            
            # Filter Sunday claims
            sunday_claims = df[
                df[col_mapping["Treatment from date"]].dt.dayofweek == 6  # Sunday = 6
            ]
            
            return sunday_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
        except Exception as e:
            raise Exception(f"Scenario 4 failed: {str(e)}")
    
    def scenario_5_invalid_invoices(self, df):
        """Scenario 5: Invalid Invoice References"""
        try:
            required_cols = ["Member ID", "Invoice No Reference", "Claim ID", "Provider ID"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Find invalid invoice references (not length 8)
            invalid_invoices = df[
                df[col_mapping["Invoice No Reference"]].astype(str).str.len() != 8
            ]
            
            # Find duplicates
            duplicates = invalid_invoices.groupby([col_mapping["Member ID"], col_mapping["Invoice No Reference"]]).agg({
                col_mapping["Claim ID"]: 'nunique'
            }).reset_index()
            
            duplicates = duplicates[duplicates[col_mapping["Claim ID"]] > 1]
            
            if not duplicates.empty:
                # Get flagged claims
                flagged = df.merge(
                    duplicates[[col_mapping["Member ID"], col_mapping["Invoice No Reference"]]],
                    on=[col_mapping["Member ID"], col_mapping["Invoice No Reference"]]
                )
                return flagged[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame()
            
        except Exception as e:
            raise Exception(f"Scenario 5 failed: {str(e)}")
    
    def scenario_6_same_day_treatments(self, df):
        """Scenario 6: Same Day Inpatient-Outpatient Treatments"""
        try:
            required_cols = ["Member ID", "Treatment from date", "Treatment to date", "Coverage type (Inpatient, Outpatient, Pharmacy, etc.)"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Find same day treatments with different coverage types
            same_day = df[
                df[col_mapping["Treatment from date"]] == df[col_mapping["Treatment to date"]]
            ]
            
            # Group by member and date to find multiple coverage types
            coverage_counts = same_day.groupby([col_mapping["Member ID"], col_mapping["Treatment from date"]]).agg({
                col_mapping["Coverage type (Inpatient, Outpatient, Pharmacy, etc.)"]: 'nunique'
            }).reset_index()
            
            suspicious = coverage_counts[
                coverage_counts[col_mapping["Coverage type (Inpatient, Outpatient, Pharmacy, etc.)"]] > 1
            ]
            
            if not suspicious.empty:
                flagged = df.merge(
                    suspicious[[col_mapping["Member ID"], col_mapping["Treatment from date"]]],
                    on=[col_mapping["Member ID"], col_mapping["Treatment from date"]]
                )
                return flagged[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame()
            
        except Exception as e:
            raise Exception(f"Scenario 6 failed: {str(e)}")
    
    def scenario_7_multi_country(self, df):
        """Scenario 7: Multi-Country Provider Analysis"""
        try:
            required_cols = ["Provider ID", "Provider_country_code", "Claim ID"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Count countries per provider
            provider_countries = df.groupby(col_mapping["Provider ID"]).agg({
                col_mapping["Provider_country_code"]: 'nunique'
            }).reset_index()
            
            # Filter providers with >3 countries
            multi_country_providers = provider_countries[
                provider_countries[col_mapping["Provider_country_code"]] > 3
            ]
            
            if not multi_country_providers.empty:
                flagged = df[df[col_mapping["Provider ID"]].isin(multi_country_providers[col_mapping["Provider ID"]])]
                return flagged[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame()
            
        except Exception as e:
            raise Exception(f"Scenario 7 failed: {str(e)}")
    
    def scenario_8_multiple_providers(self, df):
        """Scenario 8: Multiple Providers Same Day"""
        try:
            required_cols = ["Member ID", "Treatment from date", "Provider ID", "Payment currency code"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Count providers per member per day
            provider_counts = df.groupby([col_mapping["Member ID"], col_mapping["Treatment from date"]]).agg({
                col_mapping["Provider ID"]: 'nunique'
            }).reset_index()
            
            # Filter members with >2 providers same day
            multiple_providers = provider_counts[
                provider_counts[col_mapping["Provider ID"]] > 2
            ]
            
            if not multiple_providers.empty:
                flagged = df.merge(
                    multiple_providers[[col_mapping["Member ID"], col_mapping["Treatment from date"]]],
                    on=[col_mapping["Member ID"], col_mapping["Treatment from date"]]
                )
                return flagged[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame()
            
        except Exception as e:
            raise Exception(f"Scenario 8 failed: {str(e)}")

class MLScenarios:
    """Machine Learning-based FWA detection scenarios"""
    
    def __init__(self):
        self.scenarios = {
            "Isolation Forest Anomaly Detection": {
                "description": "Uses Isolation Forest to detect anomalous claims",
                "required_fields": ["Paid amount", "Age", "Incident count"],
                "function": self.isolation_forest_detection
            },
            "DBSCAN Clustering Analysis": {
                "description": "Identifies unusual claim clusters using DBSCAN",
                "required_fields": ["Paid amount", "Age", "Provider ID"],
                "function": self.dbscan_clustering
            },
            "Random Forest Fraud Prediction": {
                "description": "Predicts fraud probability using Random Forest",
                "required_fields": ["Paid amount", "Age", "Provider ID", "Diagnostic_code (ICD-10)"],
                "function": self.random_forest_prediction
            },
            "Statistical Pattern Recognition": {
                "description": "Identifies patterns using statistical methods",
                "required_fields": ["Paid amount", "Provider ID", "Member ID"],
                "function": self.pattern_recognition
            }
        }
    
    def get_available_scenarios(self):
        return self.scenarios
    
    def run_scenario(self, scenario_name, df):
        """Run a specific ML scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"ML Scenario '{scenario_name}' not found")
        
        scenario_func = self.scenarios[scenario_name]["function"]
        return scenario_func(df)
    
    def prepare_features(self, df, feature_columns):
        """Prepare features for ML algorithms"""
        try:
            # Select and clean features
            features_df = df[feature_columns].copy()
            
            # Handle missing values
            for col in features_df.columns:
                if pd.api.types.is_numeric_dtype(features_df[col]):
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
                else:
                    features_df[col] = features_df[col].astype(str).fillna('Unknown')
            
            # Encode categorical variables
            le = LabelEncoder()
            for col in features_df.columns:
                if not pd.api.types.is_numeric_dtype(features_df[col]):
                    features_df[col] = le.fit_transform(features_df[col])
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_df)
            
            return features_scaled, features_df.index
            
        except Exception as e:
            raise Exception(f"Feature preparation failed: {str(e)}")
    
    def isolation_forest_detection(self, df):
        """Isolation Forest Anomaly Detection"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Incident count']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return pd.DataFrame()
            
            features, indices = self.prepare_features(df, available_cols)
            
            # Run Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(features)
            
            # Get anomalous claims
            anomaly_indices = indices[anomaly_labels == -1]
            flagged_claims = df.loc[anomaly_indices]
            
            return flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
        except Exception as e:
            raise Exception(f"Isolation Forest failed: {str(e)}")
    
    def dbscan_clustering(self, df):
        """DBSCAN Clustering Analysis"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Provider ID']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return pd.DataFrame()
            
            features, indices = self.prepare_features(df, available_cols)
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(features)
            
            # Get outliers (label = -1)
            outlier_indices = indices[cluster_labels == -1]
            flagged_claims = df.loc[outlier_indices]
            
            return flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
        except Exception as e:
            raise Exception(f"DBSCAN failed: {str(e)}")
    
    def random_forest_prediction(self, df):
        """Random Forest Fraud Prediction"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Provider ID', 'Diagnostic_code (ICD-10)']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) < 3:
                return pd.DataFrame()
            
            features, indices = self.prepare_features(df, available_cols)
            
            # Create synthetic labels for demonstration (in real scenario, use historical fraud data)
            # High amounts and unusual patterns get higher fraud probability
            paid_amounts = pd.to_numeric(df['Paid amount'], errors='coerce').fillna(0)
            threshold = paid_amounts.quantile(0.9)
            synthetic_labels = (paid_amounts > threshold).astype(int)
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, synthetic_labels)
            
            # Predict fraud probability
            fraud_proba = rf.predict_proba(features)[:, 1]
            
            # Flag high-risk claims (top 10%)
            risk_threshold = np.percentile(fraud_proba, 90)
            high_risk_indices = indices[fraud_proba > risk_threshold]
            flagged_claims = df.loc[high_risk_indices]
            
            return flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
        except Exception as e:
            raise Exception(f"Random Forest failed: {str(e)}")
    
    def pattern_recognition(self, df):
        """Statistical Pattern Recognition"""
        try:
            feature_cols = ['Paid amount', 'Provider ID', 'Member ID']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return pd.DataFrame()
            
            # Provider pattern analysis
            if 'Provider ID' in df.columns and 'Paid amount' in df.columns:
                provider_stats = df.groupby('Provider ID').agg({
                    'Paid amount': ['mean', 'std', 'count']
                }).reset_index()
                
                provider_stats.columns = ['Provider ID', 'avg_amount', 'std_amount', 'claim_count']
                
                # Flag providers with unusual patterns
                avg_threshold = provider_stats['avg_amount'].quantile(0.95)
                count_threshold = provider_stats['claim_count'].quantile(0.95)
                
                suspicious_providers = provider_stats[
                    (provider_stats['avg_amount'] > avg_threshold) |
                    (provider_stats['claim_count'] > count_threshold)
                ]
                
                if not suspicious_providers.empty:
                    flagged = df[df['Provider ID'].isin(suspicious_providers['Provider ID'])]
                    return flagged[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame()
            
        except Exception as e:
            raise Exception(f"Pattern Recognition failed: {str(e)}")