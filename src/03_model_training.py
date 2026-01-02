import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_models(df):
    print("Training classification model...")
    
    # Select features for training
    feature_cols = [
        'title_length', 'word_count', 'avg_word_length',
        'uppercase_ratio', 'digit_count', 'special_char_count',
        'exclamation_count', 'has_all_caps_word', 'unique_word_ratio',
        'spam_keyword_count', 'has_proper_capitalization', 'reviews'
    ]
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df['quality_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    
    # Train Random Forest
    print("\n" + "="*50)
    print("Training Random Forest Classifier...")
    print("="*50)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['Low Quality', 'High Quality']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    # Save model
    joblib.dump(rf_model, 'models/rf_classifier.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    print("\n✅ Model saved to: models/rf_classifier.pkl")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Quality', 'High Quality'],
                yticklabels=['Low Quality', 'High Quality'])
    plt.title('Confusion Matrix - Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion matrix saved to: outputs/confusion_matrix.png")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance in Quality Classification')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Feature importance plot saved to: outputs/feature_importance.png")
    
    return rf_model, feature_cols

if __name__ == "__main__":
    df = pd.read_csv('data/featured_data.csv')
    model, features = train_models(df)
    print("\n✅ Model training complete!")
