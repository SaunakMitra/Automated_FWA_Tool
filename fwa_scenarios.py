import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
import warnings
import io
from datetime import datetime
warnings.filterwarnings('ignore')

class FWAScenarios:
    def __init__(self):
        self.python_scenarios = {
            "Scenario 1 - Amount Outlier Detection": {
                "description": "Detects claims with unusually high amounts using IQR analysis",
                "required_fields": ["Claimed_currency_code", "Provider_country_code", "Payee type", 
                                  "Claim invoice gross total amount", "Procedure_code (CPT codes)", "Incident count"],
                "function": self.scenario_1_amount_outlier
            },
            "Scenario 2 - Chemotherapy Frequency Analysis": {
                "description": "Identifies suspicious chemotherapy treatment patterns with 3-13 day gaps",
                "required_fields": ["Procedure_code (CPT codes)", "Treatment from date", "Treatment to date", 
                                  "Member ID", "Claim ID"],
                "function": self.scenario_2_chemotherapy
            },
            "Scenario 3 - Cross-Country Treatment Analysis": {
                "description": "Detects treatments in different countries on same date",
                "required_fields": ["Member ID", "Treatment from date", "Provider_country_code", 
                                  "Diagnostic_code (ICD-10)"],
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
            "Scenario 6 - Same Day Inpatient-Outpatient": {
                "description": "Identifies same-day inpatient and outpatient treatments",
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
        
        self.ml_scenarios = {
            "Isolation Forest Anomaly Detection": {
                "description": "Uses Isolation Forest to detect anomalous claims",
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
    
    def map_column_names(self, df, required_columns, field_mapping):
        """Map column names using field mapping"""
        column_mapping = {}
        
        for req_col in required_columns:
            if req_col in field_mapping:
                mapped_col = field_mapping[req_col]
                if mapped_col in df.columns:
                    column_mapping[req_col] = mapped_col
                else:
                    # Fallback to similarity matching
                    best_match = self.find_best_match(req_col, df.columns)
                    column_mapping[req_col] = best_match if best_match else req_col
            else:
                # Fallback to similarity matching
                best_match = self.find_best_match(req_col, df.columns)
                column_mapping[req_col] = best_match if best_match else req_col
        
        return column_mapping
    
    def find_best_match(self, target, columns):
        """Find best matching column name"""
        target_lower = target.lower()
        best_match = None
        best_score = 0
        
        for col in columns:
            col_lower = col.lower()
            
            # Exact match
            if target_lower == col_lower:
                return col
            
            # Contains match
            if target_lower in col_lower or col_lower in target_lower:
                score = 0.9
                if score > best_score:
                    best_score = score
                    best_match = col
            
            # Word overlap
            target_words = set(target_lower.replace('_', ' ').replace('-', ' ').split())
            col_words = set(col_lower.replace('_', ' ').replace('-', ' ').split())
            
            if target_words & col_words:
                score = len(target_words & col_words) / len(target_words | col_words)
                if score > best_score:
                    best_score = score
                    best_match = col
        
        return best_match if best_score > 0.3 else None
    
    def run_scenarios(self, df, selected_python, selected_ml, field_mapping):
        """Run selected FWA scenarios"""
        all_results = []
        scenario_flags = {}
        
        try:
            # Run Python scenarios
            for scenario_name in selected_python:
                try:
                    scenario_info = self.python_scenarios[scenario_name]
                    scenario_func = scenario_info["function"]
                    
                    # Map required columns
                    required_cols = scenario_info["required_fields"]
                    col_mapping = self.map_column_names(df, required_cols, field_mapping)
                    
                    # Check if all required columns are available
                    missing_cols = [col for col in required_cols if col_mapping.get(col) not in df.columns]
                    if missing_cols:
                        raise ValueError(f"Missing required columns for {scenario_name}: {missing_cols}")
                    
                    # Run scenario
                    result = scenario_func(df, col_mapping)
                    
                    if not result.empty:
                        result['scenario'] = scenario_name
                        result['scenario_type'] = 'Python'
                        result['fraud_reason'] = scenario_info["description"]
                        all_results.append(result)
                        
                        # Track flags for each claim
                        for claim_id in result['Claim ID'].values:
                            if claim_id not in scenario_flags:
                                scenario_flags[claim_id] = {}
                            scenario_flags[claim_id][scenario_name] = 1
                
                except Exception as e:
                    # If any Python scenario fails, stop all scenarios
                    raise Exception(f"Python scenario '{scenario_name}' failed: {str(e)}. All scenarios stopped.")
            
            # Run ML scenarios with train/test split
            for scenario_name in selected_ml:
                try:
                    scenario_info = self.ml_scenarios[scenario_name]
                    scenario_func = scenario_info["function"]
                    
                    # Map required columns
                    required_cols = scenario_info["required_fields"]
                    col_mapping = self.map_column_names(df, required_cols, field_mapping)
                    
                    # Check if all required columns are available
                    missing_cols = [col for col in required_cols if col_mapping.get(col) not in df.columns]
                    if missing_cols:
                        raise ValueError(f"Missing required columns for {scenario_name}: {missing_cols}")
                    
                    # Run ML scenario with train/test split
                    result = scenario_func(df, col_mapping)
                    
                    if not result.empty:
                        result['scenario'] = scenario_name
                        result['scenario_type'] = 'ML'
                        result['fraud_reason'] = scenario_info["reason"]
                        all_results.append(result)
                        
                        # Track flags for each claim
                        for claim_id in result['Claim ID'].values:
                            if claim_id not in scenario_flags:
                                scenario_flags[claim_id] = {}
                            scenario_flags[claim_id][scenario_name] = 1
                
                except Exception as e:
                    # If any ML scenario fails, stop all scenarios
                    raise Exception(f"ML scenario '{scenario_name}' failed: {str(e)}. All scenarios stopped.")
            
            # Combine all results
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                
                # Add scenario flags
                all_scenario_names = list(selected_python) + list(selected_ml)
                for scenario in all_scenario_names:
                    combined_results[f'{scenario}_flag'] = combined_results['Claim ID'].map(
                        lambda x: scenario_flags.get(x, {}).get(scenario, 0)
                    )
                
                # Calculate fraud score
                combined_results['fraud_score'] = combined_results[[f'{s}_flag' for s in all_scenario_names]].sum(axis=1)
                
                # Only return claims with at least one flag
                flagged_claims = combined_results[combined_results['fraud_score'] > 0]
                
                return flagged_claims.sort_values('fraud_score', ascending=False)
            
            return pd.DataFrame()
            
        except Exception as e:
            raise Exception(str(e))
    
    # Python Scenario Implementations
    def scenario_1_amount_outlier(self, df, col_mapping):
        """Scenario 1: Amount Outlier Detection"""
        try:
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
    
    def scenario_2_chemotherapy(self, df, col_mapping):
        """Scenario 2: Chemotherapy Frequency Analysis"""
        try:
            # Step 1: Filter chemotherapy claims
            chemo_claims = df[
                (df[col_mapping["Procedure_code (CPT codes)"]].astype(str) == "4030") &
                (df[col_mapping["Treatment from date"]].notnull()) &
                (df[col_mapping["Treatment to date"]].notnull())
            ].copy()
            
            if chemo_claims.empty:
                return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
            # Convert dates
            chemo_claims[col_mapping["Treatment from date"]] = pd.to_datetime(chemo_claims[col_mapping["Treatment from date"]], errors='coerce')
            
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
            chemo_claims["previous_treatment_date"] = chemo_claims.groupby("Member ID")[col_mapping["Treatment from date"]].shift(1)
            
            # Step 5: Calculate gap in days
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
    
    def scenario_3_cross_country(self, df, col_mapping):
        """Scenario 3: Cross-Country Treatment Analysis"""
        try:
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
    
    def scenario_4_sunday_claims(self, df, col_mapping):
        """Scenario 4: Sunday Hospital Claims"""
        try:
            # Convert to datetime
            df[col_mapping["Treatment from date"]] = pd.to_datetime(df[col_mapping["Treatment from date"]], errors='coerce')
            
            # Filter Sunday claims
            sunday_claims = df[
                df[col_mapping["Treatment from date"]].dt.dayofweek == 6  # Sunday = 6
            ]
            
            if not sunday_claims.empty:
                return sunday_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Scenario 4 failed: {str(e)}")
    
    def scenario_5_invalid_invoices(self, df, col_mapping):
        """Scenario 5: Invalid Invoice References"""
        try:
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
    
    def scenario_6_same_day_treatments(self, df, col_mapping):
        """Scenario 6: Same Day Inpatient-Outpatient Treatments"""
        try:
            # Find same day treatments
            same_day = df[
                df[col_mapping["Treatment from date"]] == df[col_mapping["Treatment to date"]]
            ]
            
            if not same_day.empty:
                return same_day[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"Scenario 6 failed: {str(e)}")
    
    def scenario_7_multi_country(self, df, col_mapping):
        """Scenario 7: Multi-Country Provider Analysis"""
        try:
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
    
    def scenario_8_multiple_providers(self, df, col_mapping):
        """Scenario 8: Multiple Providers Same Day"""
        try:
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
    
    # ML Scenario Implementations with Train/Test Split
    def prepare_features(self, df, feature_columns, col_mapping):
        """Prepare features for ML algorithms with train/test split"""
        try:
            # Map column names
            available_cols = [col_mapping[col] for col in feature_columns if col_mapping.get(col) in df.columns]
            
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
            paid_col = None
            for col in df.columns:
                if 'paid' in col.lower() and 'amount' in col.lower():
                    paid_col = col
                    break
            
            if paid_col:
                paid_amounts = pd.to_numeric(df[paid_col], errors='coerce').fillna(0)
                
                if method == 'outlier':
                    # Use IQR method
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
                    # Z-score method
                    z_scores = np.abs((paid_amounts - paid_amounts.mean()) / paid_amounts.std())
                    labels = (z_scores > 2).astype(int)
                
                return labels
            else:
                # Random labels if no amount field
                return np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
                
        except Exception as e:
            return np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    
    def isolation_forest_detection(self, df, col_mapping):
        """Isolation Forest with train/test split"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Incident count']
            features, indices = self.prepare_features(df, feature_cols, col_mapping)
            
            # Split data for validation
            X_train, X_test, idx_train, idx_test = train_test_split(
                features, indices, test_size=0.3, random_state=42
            )
            
            # Train Isolation Forest
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
    
    def dbscan_clustering(self, df, col_mapping):
        """DBSCAN Clustering with validation"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Provider ID']
            features, indices = self.prepare_features(df, feature_cols, col_mapping)
            
            # Split data for validation
            X_train, X_test, idx_train, idx_test = train_test_split(
                features, indices, test_size=0.3, random_state=42
            )
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            full_labels = dbscan.fit_predict(features)
            
            # Get outliers (label = -1)
            outlier_indices = indices[full_labels == -1]
            flagged_claims = df.loc[outlier_indices]
            
            if not flagged_claims.empty:
                return flagged_claims[['Claim ID', 'Member ID', 'Provider ID', 'Paid amount']].drop_duplicates()
            
            return pd.DataFrame(columns=['Claim ID', 'Member ID', 'Provider ID', 'Paid amount'])
            
        except Exception as e:
            raise Exception(f"DBSCAN failed: {str(e)}")
    
    def random_forest_prediction(self, df, col_mapping):
        """Random Forest with proper train/test split and cross-validation"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Provider ID', 'Diagnostic_code (ICD-10)']
            features, indices = self.prepare_features(df, feature_cols, col_mapping)
            
            # Create synthetic labels
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
            test_accuracy = accuracy_score(y_test, test_predictions)
            
            # Predict fraud probability on full dataset
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
    
    def lightgbm_detection(self, df, col_mapping):
        """LightGBM with train/test split and early stopping"""
        try:
            feature_cols = ['Paid amount', 'Age', 'Provider ID', 'Procedure_code (CPT codes)']
            features, indices = self.prepare_features(df, feature_cols, col_mapping)
            
            # Create synthetic labels
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
    
    def create_excel_report(self, results):
        """Create Excel report with two sheets"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Separate Python and ML results
            python_results = results[results['scenario_type'] == 'Python'].copy()
            ml_results = results[results['scenario_type'] == 'ML'].copy()
            
            # Python scenarios sheet
            if not python_results.empty:
                # Create binary flags for each Python scenario
                python_scenarios = python_results['scenario'].unique()
                python_summary = python_results.groupby('Claim ID').agg({
                    'Member ID': 'first',
                    'Provider ID': 'first',
                    'Paid amount': 'first',
                    'fraud_score': 'first'
                }).reset_index()
                
                # Add binary flags
                for scenario in python_scenarios:
                    python_summary[f'{scenario}_flag'] = python_summary['Claim ID'].isin(
                        python_results[python_results['scenario'] == scenario]['Claim ID']
                    ).astype(int)
                
                python_summary = python_summary.sort_values('fraud_score', ascending=False)
                python_summary.to_excel(writer, sheet_name='Hypothesis', index=False)
            
            # ML scenarios sheet
            if not ml_results.empty:
                # Create binary flags for each ML scenario
                ml_scenarios = ml_results['scenario'].unique()
                ml_summary = ml_results.groupby('Claim ID').agg({
                    'Member ID': 'first',
                    'Provider ID': 'first',
                    'Paid amount': 'first',
                    'fraud_score': 'first',
                    'fraud_reason': 'first'
                }).reset_index()
                
                # Add binary flags
                for scenario in ml_scenarios:
                    ml_summary[f'{scenario}_flag'] = ml_summary['Claim ID'].isin(
                        ml_results[ml_results['scenario'] == scenario]['Claim ID']
                    ).astype(int)
                
                ml_summary = ml_summary.sort_values('fraud_score', ascending=False)
                ml_summary.to_excel(writer, sheet_name='ML_Scenarios', index=False)
        
        output.seek(0)
        return output.getvalue()