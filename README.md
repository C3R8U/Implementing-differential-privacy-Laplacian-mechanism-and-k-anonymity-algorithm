# Implementing-differential-privacy-Laplacian-mechanism-and-k-anonymity-algorithm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class DifferentialPrivacyMechanisms:
    """Implement differential privacy mechanisms for data anonymization"""
    
    @staticmethod
    def laplace_mechanism(query_result, sensitivity, epsilon):
        """
        Apply Laplace mechanism for ε-differential privacy
        
        Args:
            query_result: Original query result (numeric)
            sensitivity: Global sensitivity of the query (Δf)
            epsilon: Privacy budget (ε)
            
        Returns:
            Noisy query result satisfying ε-DP
        """
        # Calculate scale parameter for Laplace distribution
        scale = sensitivity / epsilon
        
        # Generate noise from Laplace distribution
        noise = np.random.laplace(0, scale, 1)[0]
        
        # Return noisy result
        return query_result + noise
    
    @staticmethod
    def gaussian_mechanism(query_result, sensitivity, epsilon, delta):
        """
        Apply Gaussian mechanism for (ε,δ)-differential privacy
        
        Args:
            query_result: Original query result (numeric)
            sensitivity: Global sensitivity of the query (Δf)
            epsilon: Privacy budget (ε)
            delta: Privacy parameter (δ), typically very small (e.g., 1e-5)
            
        Returns:
            Noisy query result satisfying (ε,δ)-DP
        """
        # Calculate standard deviation for Gaussian noise
        sigma = sensitivity * np.sqrt(2 * np.log(1.25/delta)) / epsilon
        
        # Generate noise from Gaussian distribution
        noise = np.random.normal(0, sigma, 1)[0]
        
        return query_result + noise

class KAnonymity:
    """Implement k-anonymity through generalization and suppression"""
    
    def __init__(self, k, quasi_identifiers):
        """
        Initialize k-anonymity parameters
        
        Args:
            k: Anonymity parameter (minimum group size)
            quasi_identifiers: List of column names considered as QIs
        """
        self.k = k
        self.quasi_identifiers = quasi_identifiers
        self.generalization_hierarchies = {}
    
    def create_generalization_hierarchy(self, column, hierarchy):
        """
        Create generalization hierarchy for a column
        
        Args:
            column: Column name to create hierarchy for
            hierarchy: Dictionary mapping specific values to generalized values
        """
        self.generalization_hierarchies[column] = hierarchy
    
    def calculate_ncp(self, original_data, anonymized_data, column):
        """
        Calculate Normalized Certainty Penalty (NCP) for a column
        
        Args:
            original_data: Original DataFrame
            anonymized_data: Anonymized DataFrame
            column: Column name to calculate NCP for
            
        Returns:
            NCP value between 0 and 1
        """
        if column not in self.generalization_hierarchies:
            return 0.0  # No generalization applied
        
        original_unique = len(original_data[column].unique())
        anonymized_unique = len(anonymized_data[column].unique())
        
        if original_unique == 0:
            return 0.0
        
        return 1.0 - (anonymized_unique / original_unique)
    
    def apply_k_anonymity(self, data):
        """
        Apply k-anonymity to the dataset using generalization
        
        Args:
            data: Input DataFrame containing sensitive data
            
        Returns:
            Anonymized DataFrame satisfying k-anonymity
        """
        anonymized_data = data.copy()
        
        # Apply generalization to quasi-identifiers
        for column in self.quasi_identifiers:
            if column in self.generalization_hierarchies:
                hierarchy = self.generalization_hierarchies[column]
                anonymized_data[column] = anonymized_data[column].map(
                    lambda x: hierarchy.get(x, x)
                )
        
        # Ensure k-anonymity by grouping
        group_sizes = anonymized_data.groupby(self.quasi_identifiers).size()
        valid_groups = group_sizes[group_sizes >= self.k]
        
        # Filter data to only include records in valid k-anonymous groups
        valid_indices = []
        for group_values, count in valid_groups.items():
            if count >= self.k:
                mask = True
                for i, col in enumerate(self.quasi_identifiers):
                    mask = mask & (anonymized_data[col] == group_values[i])
                valid_indices.extend(anonymized_data[mask].index.tolist())
        
        return anonymized_data.loc[valid_indices].reset_index(drop=True)

def evaluate_utility(original_data, anonymized_data, target_column):
    """
    Evaluate utility by comparing machine learning model performance
    
    Args:
        original_data: Original dataset
        anonymized_data: Anonymized dataset
        target_column: Name of the target variable for classification
        
    Returns:
        Dictionary containing AUC scores for both datasets
    """
    # Prepare features and target
    features = [col for col in original_data.columns if col != target_column]
    
    X_orig = original_data[features]
    y_orig = original_data[target_column]
    
    X_anon = anonymized_data[features]
    y_anon = anonymized_data[target_column]
    
    # Convert categorical variables to numerical
    X_orig = pd.get_dummies(X_orig)
    X_anon = pd.get_dummies(X_anon)
    
    # Align columns (anonymized data might have different columns due to generalization)
    common_cols = X_orig.columns.intersection(X_anon.columns)
    X_orig = X_orig[common_cols]
    X_anon = X_anon[common_cols]
    
    # Split data
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
        X_orig, y_orig, test_size=0.3, random_state=42
    )
    
    X_anon_train, X_anon_test, y_anon_train, y_anon_test = train_test_split(
        X_anon, y_anon, test_size=0.3, random_state=42
    )
    
    # Train model on original data
    model_orig = RandomForestClassifier(n_estimators=100, random_state=42)
    model_orig.fit(X_orig_train, y_orig_train)
    y_pred_orig = model_orig.predict_proba(X_orig_test)[:, 1]
    auc_orig = roc_auc_score(y_orig_test, y_pred_orig)
    
    # Train model on anonymized data
    model_anon = RandomForestClassifier(n_estimators=100, random_state=42)
    model_anon.fit(X_anon_train, y_anon_train)
    y_pred_anon = model_anon.predict_proba(X_anon_test)[:, 1]
    auc_anon = roc_auc_score(y_anon_test, y_pred_anon)
    
    return {
        'original_auc': auc_orig,
        'anonymized_auc': auc_anon,
        'auc_loss': auc_orig - auc_anon
    }

def main():
    """Main function to demonstrate the anonymization techniques"""
    
    # Create sample medical dataset (simulating MIMIC-III subset)
    np.random.seed(42)
    n_records = 1000
    
    # Generate synthetic data with realistic distributions
    data = pd.DataFrame({
        'age': np.random.randint(18, 90, n_records),
        'zip_code': np.random.choice(['10001', '10002', '10003', '10004', '10005'], n_records),
        'diagnosis_code': np.random.choice(['A01', 'A02', 'B01', 'B02', 'C01'], n_records),
        'lab_result': np.random.normal(100, 15, n_records),
        'disease_risk': np.random.choice([0, 1], n_records, p=[0.7, 0.3])  # 30% have high risk
    })
    
    print("Original Dataset Summary:")
    print(f"Number of records: {len(data)}")
    print(f"Disease prevalence: {data['disease_risk'].mean():.3f}")
    print("\nFirst 5 records:")
    print(data.head())
    
    # Demonstrate Differential Privacy
    print("\n" + "="*50)
    print("DIFFERENTIAL PRIVACY DEMONSTRATION")
    print("="*50)
    
    dp = DifferentialPrivacyMechanisms()
    
    # Example 1: Count query with Laplace mechanism
    patient_count = len(data)
    sensitivity_count = 1  # Adding/removing one record changes count by 1
    epsilon = 1.0
    
    noisy_count = dp.laplace_mechanism(patient_count, sensitivity_count, epsilon)
    print(f"Original patient count: {patient_count}")
    print(f"Noisy count (ε={epsilon}): {noisy_count:.1f}")
    print(f"Relative error: {abs(noisy_count - patient_count)/patient_count*100:.1f}%")
    
    # Example 2: Average query with Gaussian mechanism
    avg_lab_result = data['lab_result'].mean()
    sensitivity_avg = 200/len(data)  # Assuming lab result range is 0-200
    delta = 1e-5
    
    noisy_avg = dp.gaussian_mechanism(avg_lab_result, sensitivity_avg, epsilon, delta)
    print(f"\nOriginal average lab result: {avg_lab_result:.2f}")
    print(f"Noisy average (ε={epsilon}, δ={delta}): {noisy_avg:.2f}")
    print(f"Relative error: {abs(noisy_avg - avg_lab_result)/avg_lab_result*100:.1f}%")
    
    # Demonstrate k-Anonymity
    print("\n" + "="*50)
    print("K-ANONYMITY DEMONSTRATION")
    print("="*50)
    
    # Define quasi-identifiers and create generalization hierarchies
    quasi_identifiers = ['age', 'zip_code', 'diagnosis_code']
    
    k_anon = KAnonymity(k=5, quasi_identifiers=quasi_identifiers)
    
    # Create age generalization hierarchy
    age_hierarchy = {}
    for age in range(18, 91):
        if age < 30:
            age_hierarchy[age] = '20-29'
        elif age < 40:
            age_hierarchy[age] = '30-39'
        elif age < 50:
            age_hierarchy[age] = '40-49'
        elif age < 60:
            age_hierarchy[age] = '50-59'
        else:
            age_hierarchy[age] = '60+'
    
    k_anon.create_generalization_hierarchy('age', age_hierarchy)
    
    # Create diagnosis code generalization (group by first letter)
    diag_hierarchy = {
        'A01': 'A', 'A02': 'A',
        'B01': 'B', 'B02': 'B',
        'C01': 'C'
    }
    k_anon.create_generalization_hierarchy('diagnosis_code', diag_hierarchy)
    
    # Apply k-anonymity
    anonymized_data = k_anon.apply_k_anonymity(data)
    
    print(f"Original dataset size: {len(data)} records")
    print(f"After k-anonymity (k=5): {len(anonymized_data)} records")
    print(f"Records retained: {len(anonymized_data)/len(data)*100:.1f}%")
    
    # Calculate NCP for each quasi-identifier
    print("\nNormalized Certainty Penalty (NCP) by attribute:")
    for column in quasi_identifiers:
        ncp = k_anon.calculate_ncp(data, anonymized_data, column)
        print(f"  {column}: {ncp:.3f}")
    
    # Evaluate utility through machine learning performance
    print("\n" + "="*50)
    print("UTILITY EVALUATION (Machine Learning Performance)")
    print("="*50)
    
    utility_results = evaluate_utility(data, anonymized_data, 'disease_risk')
    
    print(f"AUC on original data: {utility_results['original_auc']:.3f}")
    print(f"AUC on anonymized data: {utility_results['anonymized_auc']:.3f}")
    print(f"AUC loss: {utility_results['auc_loss']:.3f}")
    
    # Visualization of data utility comparison
    plt.figure(figsize=(10, 6))
    
    categories = ['Original Data', 'Anonymized Data']
    auc_values = [utility_results['original_auc'], utility_results['anonymized_auc']]
    
    plt.bar(categories, auc_values, color=['blue', 'orange'], alpha=0.7)
    plt.ylabel('AUC Score')
    plt.title('Model Performance: Original vs Anonymized Data')
    plt.ylim(0, 1.0)
    
    for i, v in enumerate(auc_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDemonstration completed successfully!")
    print("This code implements the core anonymization techniques discussed in the paper.")
    print("Key features demonstrated:")
    print("1. Differential Privacy: Laplace and Gaussian mechanisms")
    print("2. k-Anonymity: Generalization and suppression techniques") 
    print("3. Utility evaluation: Machine learning performance comparison")
    print("4. Privacy-utility tradeoff: Quantitative analysis of the balance")

if __name__ == "__main__":
    main()
