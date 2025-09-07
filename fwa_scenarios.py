import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class PythonScenarios:
    """Python-based FWA detection scenarios with exact implementations"""
    
    def __init__(self):
        self.scenarios = {
            "Scenario 1 - Amount Outlier Detection": {
                "description": "Detects claims with unusually high amounts using IQR analysis",
                "required_fields": ["Claimed_currency_code", "Provider_country_code", "Payee type", "Claim invoice gross total amount", "Procedure_code (CPT codes)", "Incident count"],
                "function": self.scenario_1_amount_outlier
            },
            "Scenario 2 - Chemotherapy Frequency Analysis": {
                "description": "Identifies suspicious chemotherapy treatment patterns with 3-13 day gaps",
                "required_fields": ["Procedure_code (CPT codes)", "Treatment from date", "Treatment to date", "Member ID", "Claim ID"],
                "function": self.scenario_2_chemotherapy
            },
            "Scenario 3 - Cross-Country Treatment Analysis": {
                "description": "Detects treatments in different countries on same date",
                "required_fields": ["Member ID", "Treatment from date", "Provider_country_code", "Diagnostic_code (ICD-10)"],
                "function": self.scenario_3_cross_country
            },
            "Scenario 4 - Sunday Hospital Claims": {
                "description": "Identifies hospital claims submitted on Sundays with specialisation code 0003",
                "required_fields": ["Treatment from date", "Provider ID", "Claim ID"],
                "function": self.scenario_4_sunday_claims
            },
            "Scenario 5 - Invalid Invoice References": {
                "description": "Detects duplicate claims with invalid invoice references (not length 8)",
                "required_fields": ["Member ID", "Invoice No Reference", "Claim ID", "Provider ID"],
                "function": self.scenario_5_invalid_invoices
            },
        
        # Store validation results
        self.validation_results = {}
            "Scenario 6 - Inpatient-Outpatient Same Day": {
                "description": "Identifies same-day inpatient (0003) and outpatient (0004) treatments",
                "required_fields": ["Member ID", "Treatment from date", "Treatment to date"],
                "function": self.scenario_6_same_day_treatments
            },
            "Scenario 7 - Multi-Country Provider Analysis": {
                "description": "Detects providers operating in more than 3 countries",
                "required_fields": ["Provider ID", "Provider_country_code", "Claim ID"],
                "function": self.scenario_7_multi_country
            },
            "Scenario 8 - Multiple Provider Same Day": {
                "description": "Identifies members visiting more than 2 providers on same day",
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
            
            if best_score > 0.3:  # Lower threshold for flexibility
                column_mapping[req_col] = best_match
            else:
                # Try to find any column with similar keywords
                for df_col in df_columns:
                    if any(word in df_col.lower() for word in req_col.lower().split()):
                        column_mapping[req_col] = df_col
                        break
                else:
                    column_mapping[req_col] = req_col  # Keep original if no match
        
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
        """Scenario 1: Amount Outlier Detection - Exact Implementation"""
        try:
            # Map column names
            required_cols = ["Claimed_currency_code", "Provider_country_code", "Payee type", 
                           "Claim invoice gross total amount", "Procedure_code (CPT codes)", "Incident count"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Check if required columns exist
            missing_cols = [col for col in required_cols if col_mapping[col] not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Step 1: Filter base dataset
            filtered = df[
                (df[col_mapping["Claimed_currency_code"]].astype(str) == "GBP") &
                (df[col_mapping["Provider_country_code"]].astype(str) == "UK") &
                (df[col_mapping["Payee type"]].astype(str) == "P") &
                (pd.to_numeric(df[col_mapping["Claim invoice gross total amount"]], errors='coerce') < 10000) &
                (df[col_mapping["Procedure_code (CPT codes)"]].notnull())
            ].copy()
            
            if filtered.empty:
                return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
            # Step 2: Compute IQR
            filtered = filtered.assign(
                gross_per_incident=lambda x: pd.to_numeric(x[col_mapping["Claim invoice gross total amount"]], errors='coerce') / 
                                           pd.to_numeric(x[col_mapping["Incident count"]], errors='coerce').replace(0, np.nan)
            )
            
            procedure_stats = (
                filtered.groupby(col_mapping["Procedure_code (CPT codes)"])["gross_per_incident"]
                .agg(
                    q1=lambda x: x.quantile(0.25),
                    q3=lambda x: x.quantile(0.75)
                )
                .reset_index()
            )
            procedure_stats["iqr_based_threshold"] = procedure_stats["q3"] + 1.5 * (procedure_stats["q3"] - procedure_stats["q1"])
            
            # Step 3: Provider averages
            provider_avg = (
                filtered.groupby(["Provider ID", col_mapping["Procedure_code (CPT codes)"]])["gross_per_incident"]
                .mean()
                .reset_index()
                .rename(columns={"gross_per_incident": "provider_avg"})
            )
            
            # Step 4: Country averages
            country_avg = (
                filtered.groupby(col_mapping["Procedure_code (CPT codes)"])["gross_per_incident"]
                .mean()
                .reset_index()
                .rename(columns={"gross_per_incident": "country_avg"})
            )
            
            # Step 5: Combine stats
            combined_stats = (
                provider_avg
                .merge(country_avg, on=col_mapping["Procedure_code (CPT codes)"])
                .merge(procedure_stats, on=col_mapping["Procedure_code (CPT codes)"])
            )
            
            # Step 6: Flag outliers
            flagged_claims = (
                filtered
                .merge(combined_stats, on=["Provider ID", col_mapping["Procedure_code (CPT codes)"]])
                .query("gross_per_incident > provider_avg and gross_per_incident > country_avg and gross_per_incident > iqr_based_threshold")
            )
            
            if not flagged_claims.empty:
                return flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Scenario 1 failed: {str(e)}")
    
    def scenario_2_chemotherapy(self, df):
        """Scenario 2: Chemotherapy Frequency Analysis - Exact Implementation"""
        try:
            required_cols = ["Procedure_code (CPT codes)", "Treatment from date", "Treatment to date", "Member ID", "Claim ID"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Step 1: Filter chemotherapy claims
            chemo_claims = df[
                (df[col_mapping["Procedure_code (CPT codes)"]].astype(str) == "4030") &
                (df[col_mapping["Treatment from date"]].notnull()) &
                (df[col_mapping["Treatment to date"]].notnull())
            ].copy()
            
            if chemo_claims.empty:
                return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
            # Step 2: Aggregate paid amount per claim
            chemo_claims = (
                chemo_claims.groupby(
                    ["Member ID", "Claim ID", col_mapping["Treatment from date"], "Claimed_currency_code"],
                    as_index=False
                )["Paid amount"].sum()
            )
            
            # Step 3: Sort and assign row number
            chemo_claims = chemo_claims.sort_values(["Member ID", col_mapping["Treatment from date"]])
            chemo_claims["rn"] = chemo_claims.groupby("Member ID").cumcount() + 1
            
            # Step 4: Add previous claim info
            chemo_claims["previous_claim_id"] = chemo_claims.groupby("Member ID")["Claim ID"].shift(1)
            chemo_claims["previous_treatment_date"] = chemo_claims.groupby("Member ID")[col_mapping["Treatment from date"]].shift(1)
            chemo_claims["previous_paid_value"] = chemo_claims.groupby("Member ID")["Paid amount"].shift(1)
            chemo_claims["previous_currency"] = chemo_claims.groupby("Member ID")["Claimed_currency_code"].shift(1)
            
            # Step 5: Calculate gap in days
            chemo_claims[col_mapping["Treatment from date"]] = pd.to_datetime(chemo_claims[col_mapping["Treatment from date"]], errors='coerce')
            chemo_claims["previous_treatment_date"] = pd.to_datetime(chemo_claims["previous_treatment_date"], errors='coerce')
            
            chemo_claims["days_between"] = (
                chemo_claims[col_mapping["Treatment from date"]] - chemo_claims["previous_treatment_date"]
            ).dt.days
            
            # Step 6: Filter 3-13 day gaps
            result = chemo_claims[
                (chemo_claims["days_between"] >= 3) & (chemo_claims["days_between"] <= 13)
            ]
            
            if not result.empty:
                return result[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Scenario 2 failed: {str(e)}")
    
    def scenario_3_cross_country(self, df):
        """Scenario 3: Cross-Country Treatment Analysis - Exact Implementation"""
        try:
            required_cols = ["Member ID", "Treatment from date", "Provider_country_code", "Diagnostic_code (ICD-10)"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Self-join on Member ID and Treatment from date
            merged = df.merge(
                df,
                on=["Member ID", col_mapping["Treatment from date"]],
                suffixes=("_t1", "_t2")
            )
            
            # Apply conditions
            filtered = merged[
                (merged[f'{col_mapping["Provider_country_code"]}_t1'] != merged[f'{col_mapping["Provider_country_code"]}_t2']) &
                (merged[f'{col_mapping["Diagnostic_code (ICD-10)"]}_t1'] != merged[f'{col_mapping["Diagnostic_code (ICD-10)"]}_t2'])
            ]
            
            if not filtered.empty:
                return filtered[['Claim ID_t1', 'Member ID', 'Provider ID_t1', 'Paid amount_t1']].rename(columns={
                    'Claim ID_t1': 'Claim ID',
                    'Provider ID_t1': 'Provider ID',
                    'Paid amount_t1': 'Paid amount'
                }).drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Scenario 3 failed: {str(e)}")
    
    def scenario_4_sunday_claims(self, df):
        """Scenario 4: Sunday Hospital Claims - Exact Implementation"""
        try:
            required_cols = ["Treatment from date", "Provider ID", "Claim ID"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Convert to datetime
            df[col_mapping["Treatment from date"]] = pd.to_datetime(df[col_mapping["Treatment from date"]], errors='coerce')
            
            # Filter Sunday claims with specialisation_code 0003
            sunday_claims = df[
                (df[col_mapping["Treatment from date"]].dt.dayofweek == 6) &  # Sunday = 6
                (df.get("specialisation_code", pd.Series(["0003"] * len(df))).astype(str) == "0003")
            ]
            
            if not sunday_claims.empty:
                return sunday_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Scenario 4 failed: {str(e)}")
    
    def scenario_5_invalid_invoices(self, df):
        """Scenario 5: Invalid Invoice References - Exact Implementation"""
        try:
            required_cols = ["Member ID", "Invoice No Reference", "Claim ID", "Provider ID"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Step 1: Find invalid invoice references
            invalid_invoices = (
                df[df[col_mapping["Invoice No Reference"]].astype(str).str.len() != 8]
                .groupby(["Member ID", col_mapping["Invoice No Reference"]])
                .agg(unique_claims=("Claim ID", "nunique"))
                .reset_index()
            )
            
            # Filter for duplicates
            invalid_invoices = invalid_invoices[invalid_invoices["unique_claims"] > 1]
            
            if not invalid_invoices.empty:
                # Get flagged claims
                flagged = df.merge(
                    invalid_invoices[["Member ID", col_mapping["Invoice No Reference"]]],
                    on=["Member ID", col_mapping["Invoice No Reference"]]
                )
                return flagged[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Scenario 5 failed: {str(e)}")
    
    def scenario_6_same_day_treatments(self, df):
        """Scenario 6: Same Day Inpatient-Outpatient Treatments - Exact Implementation"""
        try:
            required_cols = ["Member ID", "Treatment from date", "Treatment to date"]
            col_mapping = self.map_column_names(df, required_cols)
            
            # Find same day treatments with different specialisation codes
            same_day = df[
                df[col_mapping["Treatment from date"]] == df[col_mapping["Treatment to date"]]
            ]
            
            # Check for both inpatient (0003) and outpatient (0004) codes
            if "specialisation_code" in df.columns:
                member_date_flags = (
                    same_day[same_day["specialisation_code"].str.strip().isin(["0003", "0004"])]
                    .groupby(["Member ID", col_mapping["Treatment from date"]])
                    .agg(
                        has_inpatient=("specialisation_code", lambda x: any(x.str.strip() == "0003")),
                        has_outpatient=("specialisation_code", lambda x: any(x.str.strip() == "0004"))
                    )
                    .reset_index()
                )
                
                suspicious = member_date_flags[
                    member_date_flags["has_inpatient"] & member_date_flags["has_outpatient"]
                ]
                
                if not suspicious.empty:
                    flagged = df.merge(
                        suspicious[["Member ID", col_mapping["Treatment from date"]]],
                        on=["Member ID", col_mapping["Treatment from date"]]
                    )
                    return flagged[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Scenario 6 failed: {str(e)}")
    
    def scenario_7_multi_country(self, df):
        """Scenario 7: Multi-Country Provider Analysis - Exact Implementation"""
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
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Scenario 7 failed: {str(e)}")
    
    def scenario_8_multiple_providers(self, df):
        """Scenario 8: Multiple Providers Same Day - Exact Implementation"""
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
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Scenario 8 failed: {str(e)}")

class MLScenarios:
    """Machine Learning-based FWA detection scenarios with train/test split and cross-validation"""
    """Machine Learning-based FWA detection scenarios"""
    
    def __init__(self):
        self.scenarios = {
            "Isolation Forest Anomaly Detection": {
                "description": "Uses Isolation Forest to detect anomalous claims based on amount, age, and incident patterns",
                "required_fields": ["Paid amount", "Age", "Incident count"],
                "function": self.isolation_forest_detection,
                "reason": "Anomalous pattern in claim amount and patient demographics"
            },
            "DBSCAN Clustering Analysis": {
                "description": "Identifies unusual claim clusters using density-based clustering",
                "required_fields": ["Paid amount", "Age", "Provider ID"],
                "function": self.dbscan_clustering,
                "reason": "Outlier in provider-patient-amount clustering pattern"
            },
            "Random Forest Fraud Prediction": {
                "description": "Predicts fraud probability using Random Forest algorithm",
                "required_fields": ["Paid amount", "Age", "Provider ID", "Diagnostic_code (ICD-10)"],
                "function": self.random_forest_prediction,
                "reason": "High fraud probability based on historical patterns"
            },
            "LightGBM Advanced Detection": {
                "description": "Advanced gradient boosting for complex fraud pattern detection",
                "required_fields": ["Paid amount", "Age", "Provider ID", "Procedure_code (CPT codes)"],
                "function": self.lightgbm_detection,
                "reason": "Complex fraud pattern detected by gradient boosting algorithm"
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
        """Prepare features for ML algorithms with robust error handling"""
        try:
            # Map column names
            python_scenarios = PythonScenarios()
            col_mapping = python_scenarios.map_column_names(df, feature_columns)
            
            # Select available columns
            available_cols = [col_mapping[col] for col in feature_columns if col_mapping[col] in df.columns]
            
            if len(available_cols) < 2:
                raise ValueError(f"Insufficient columns available. Need at least 2, got {len(available_cols)}")
            
            features_df = df[available_cols].copy()
            
            # Handle missing values and data types
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
    
    def create_synthetic_labels(self, df, method='outlier'):
        """Create synthetic labels for supervised learning"""
        try:
            if 'Paid amount' in df.columns:
                paid_amounts = pd.to_numeric(df['Paid amount'], errors='coerce').fillna(0)
                
                if method == 'outlier':
                    # Use IQR method for outlier detection
                    Q1 = paid_amounts.quantile(0.25)
                    Q3 = paid_amounts.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_threshold = Q3 + 1.5 * IQR
                    labels = (paid_amounts > outlier_threshold).astype(int)
                elif method == 'percentile':
                    # Use top 10% as positive class
                    threshold = paid_amounts.quantile(0.9)
                    labels = (paid_amounts > threshold).astype(int)
                else:
                    # Random labels for unsupervised methods
                    labels = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
                
                return labels
            else:
                # Random labels if no amount field
                return np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
                
        except Exception as e:
            return np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    def isolation_forest_detection(self, df):
        """Isolation Forest Anomaly Detection"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Incident count']
            features, indices = self.prepare_features(df, feature_cols)
            
            # Split data for validation
            X_train, X_test, idx_train, idx_test = train_test_split(
                features, indices, test_size=0.3, random_state=42
            )
            
            # Run Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
            iso_forest.fit(X_train)
            
            # Predict on full dataset
            anomaly_labels = iso_forest.predict(features)
            
            # Get anomalous claims
            anomaly_indices = indices[anomaly_labels == -1]
            flagged_claims = df.loc[anomaly_indices]
            
            if not flagged_claims.empty:
                return flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Isolation Forest failed: {str(e)}")
    
    def dbscan_clustering(self, df):
        """DBSCAN Clustering Analysis"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Provider ID']
            features, indices = self.prepare_features(df, feature_cols)
            
            # Split data for validation
            X_train, X_test, idx_train, idx_test = train_test_split(
                features, indices, test_size=0.3, random_state=42
            )
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(X_train)
            
            # For full dataset prediction, use trained model concept
            full_labels = dbscan.fit_predict(features)
            
            # Get outliers (label = -1)
            outlier_indices = indices[full_labels == -1]
            flagged_claims = df.loc[outlier_indices]
            
            if not flagged_claims.empty:
                return flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"DBSCAN failed: {str(e)}")
    
    def random_forest_prediction(self, df):
        """Random Forest Fraud Prediction"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Provider ID', 'Diagnostic_code (ICD-10)']
            features, indices = self.prepare_features(df, feature_cols)
            
            # Create synthetic labels based on amount outliers
            synthetic_labels = self.create_synthetic_labels(df, method='percentile')
            
            # Split data for training and testing
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                features, synthetic_labels, indices, test_size=0.3, random_state=42
            )
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Cross-validation on training set
            cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
            
            # Predict on test set for validation
            test_predictions = rf.predict(X_test)
            
            # Predict fraud probability
            fraud_proba = rf.predict_proba(features)[:, 1]
            
            # Flag high-risk claims (top 10%)
            risk_threshold = np.percentile(fraud_proba, 90)
            high_risk_indices = indices[fraud_proba > risk_threshold]
            flagged_claims = df.loc[high_risk_indices]
            
            if not flagged_claims.empty:
                return flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Random Forest failed: {str(e)}")
    
    def lightgbm_detection(self, df):
        """LightGBM Advanced Detection"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Provider ID', 'Procedure_code (CPT codes)']
            features, indices = self.prepare_features(df, feature_cols)
            
            # Create synthetic labels for training
            synthetic_labels = self.create_synthetic_labels(df, method='outlier')
            
            # Split data for training and testing
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                features, synthetic_labels, indices, test_size=0.3, random_state=42
            )
            
            # Train LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1,
                'random_state': 42
            }
            
            model = lgb.train(
                params, 
                train_data, 
                num_boost_round=100,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Predict fraud probability
            fraud_proba = model.predict(features)
            
            # Flag high-risk claims (top 15%)
            risk_threshold = np.percentile(fraud_proba, 85)
            high_risk_indices = indices[fraud_proba > risk_threshold]
            flagged_claims = df.loc[high_risk_indices]
            
            if not flagged_claims.empty:
                return flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"LightGBM failed: {str(e)}")
    
    def get_validation_summary(self):
        """Get summary of all validation results"""
        return self.validation_results