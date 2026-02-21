#!/usr/bin/env python3
"""
Spotify Music Success Prediction Analysis
Predicting chart success using musical features and machine learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(filepath):
    """Load Spotify dataset"""
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """Clean and preprocess data"""
    # Handle missing values
    print(f"\nMissing values before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    df = df.dropna()
    print(f"Dataset after cleaning: {df.shape[0]} rows")
    
    # Rename columns
    rename_cols = {
        'danceability_%': 'danceability',
        'valence_%': 'valence',
        'energy_%': 'energy',
        'acousticness_%': 'acousticness',
        'instrumentalness_%': 'instrumentalness',
        'liveness_%': 'liveness',
        'speechiness_%': 'speechiness'
    }
    df = df.rename(columns=rename_cols)
    
    # Create binary target variable (top 10% = hit)
    threshold = df['streams'].quantile(0.90)
    df['is_hit'] = (df['streams'] >= threshold).astype(int)
    print(f"\nHit songs (top 10%): {df['is_hit'].sum()}")
    print(f"Non-hit songs: {(df['is_hit'] == 0).sum()}")
    
    return df

def engineer_features(df):
    """Create one-hot encoded features"""
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['key', 'mode'], prefix=['key', 'mode'])
    return df_encoded

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

def analyze_top_vs_bottom(df, n=10):
    """Compare top N vs bottom N songs"""
    top_n = df.nlargest(n, 'streams')
    bottom_n = df.nsmallest(n, 'streams')
    
    print(f"\n{'='*60}")
    print(f"TOP {n} vs BOTTOM {n} ANALYSIS")
    print(f"{'='*60}")
    
    # Mode analysis
    top_major = (top_n['mode'] == 'Major').sum() / n * 100
    bottom_major = (bottom_n['mode'] == 'Major').sum() / n * 100
    overall_major = (df['mode'] == 'Major').sum() / len(df) * 100
    
    print(f"\nMode (Major %):")
    print(f"  Top {n}: {top_major:.1f}%")
    print(f"  Bottom {n}: {bottom_major:.1f}%")
    print(f"  Overall: {overall_major:.1f}%")
    
    # Key analysis (C#)
    top_csharp = (top_n['key'] == 'C#').sum() / n * 100
    bottom_csharp = (bottom_n['key'] == 'C#').sum() / n * 100
    overall_csharp = (df['key'] == 'C#').sum() / len(df) * 100
    
    print(f"\nKey C# (%):")
    print(f"  Top {n}: {top_csharp:.1f}%")
    print(f"  Bottom {n}: {bottom_csharp:.1f}%")
    print(f"  Overall: {overall_csharp:.1f}%")
    
    # BPM analysis
    print(f"\nBPM Statistics:")
    print(f"  Top {n} - Mean: {top_n['bpm'].mean():.1f}, Std: {top_n['bpm'].std():.1f}, Min: {top_n['bpm'].min():.0f}")
    print(f"  Bottom {n} - Mean: {bottom_n['bpm'].mean():.1f}, Std: {bottom_n['bpm'].std():.1f}")
    print(f"  Overall - Mean: {df['bpm'].mean():.1f}")
    
    # Energy analysis
    print(f"\nEnergy Statistics:")
    print(f"  Top {n} - Mean: {top_n['energy'].mean():.1f}, Range: [{top_n['energy'].min():.0f}, {top_n['energy'].max():.0f}]")
    print(f"  Overall - Mean: {df['energy'].mean():.1f}")
    
    return top_n, bottom_n

def plot_distributions(df):
    """Plot feature distributions"""
    features = ['bpm', 'danceability', 'valence', 'energy', 
                'acousticness', 'instrumentalness', 'liveness', 'speechiness']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'{feature.capitalize()} Distribution')
        axes[i].set_xlabel(feature)
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: feature_distributions.png")
    plt.close()

def plot_correlation_matrix(df):
    """Plot correlation heatmap"""
    features = ['bpm', 'danceability', 'valence', 'energy', 
                'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'streams']
    
    corr_matrix = df[features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_matrix.png")
    plt.close()

def plot_key_analysis(df):
    """Analyze and plot key distribution"""
    key_counts = df['key'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    key_counts.plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Distribution of Musical Keys')
    axes[0].set_xlabel('Key')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Box plot: Key vs Streams
    top_keys = key_counts.head(8).index
    df_top_keys = df[df['key'].isin(top_keys)]
    sns.boxplot(data=df_top_keys, x='key', y='streams', ax=axes[1], palette='Set2')
    axes[1].set_title('Streams by Key (Top 8 Keys)')
    axes[1].set_xlabel('Key')
    axes[1].set_ylabel('Streams')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('key_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: key_analysis.png")
    plt.close()

def plot_mode_analysis(df):
    """Analyze mode distribution across tiers"""
    tiers = [10, 50, 100]
    major_pcts = []
    
    for n in tiers:
        top_n = df.nlargest(n, 'streams')
        major_pct = (top_n['mode'] == 'Major').sum() / n * 100
        major_pcts.append(major_pct)
    
    overall_major = (df['mode'] == 'Major').sum() / len(df) * 100
    
    plt.figure(figsize=(10, 6))
    x = ['Top 10', 'Top 50', 'Top 100', 'Overall']
    y = major_pcts + [overall_major]
    
    bars = plt.bar(x, y, color=['#2ecc71', '#3498db', '#9b59b6', '#95a5a6'])
    plt.axhline(y=overall_major, color='red', linestyle='--', label=f'Overall Average ({overall_major:.1f}%)')
    plt.title('Major Mode Percentage Across Success Tiers')
    plt.ylabel('Major Mode %')
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('mode_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: mode_analysis.png")
    plt.close()

# ============================================================================
# 3. MACHINE LEARNING MODELS
# ============================================================================

def prepare_model_data(df):
    """Prepare features and target for modeling"""
    # Select features
    feature_cols = ['bpm', 'danceability', 'valence', 'energy', 
                    'acousticness', 'instrumentalness', 'liveness', 'speechiness']
    
    # Add one-hot encoded columns
    key_cols = [col for col in df.columns if col.startswith('key_')]
    mode_cols = [col for col in df.columns if col.startswith('mode_')]
    feature_cols.extend(key_cols + mode_cols)
    
    X = df[feature_cols]
    y = df['is_hit']
    
    return X, y, feature_cols

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression"""
    print(f"\n{'='*60}")
    print("LOGISTIC REGRESSION MODEL")
    print(f"{'='*60}")
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC-ROC: {auc:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"\n5-Fold CV F1-Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    return model, y_pred

def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate Random Forest"""
    print(f"\n{'='*60}")
    print("RANDOM FOREST CLASSIFIER")
    print(f"{'='*60}")
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                   min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC-ROC: {auc:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"\n5-Fold CV F1-Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    print("\nTop 10 Feature Importances:")
    for i, idx in enumerate(indices, 1):
        print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), importances[indices])
    plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: feature_importance.png")
    plt.close()
    
    return model, y_pred

def plot_confusion_matrices(y_test, y_pred_lr, y_pred_rf):
    """Plot confusion matrices for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Logistic Regression
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Logistic Regression')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Random Forest')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrices.png")
    plt.close()

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*60)
    print("SPOTIFY MUSIC SUCCESS PREDICTION ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data('spotify-2023.csv')
    
    # Preprocess
    df = preprocess_data(df)
    
    # EDA
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    analyze_top_vs_bottom(df, n=10)
    analyze_top_vs_bottom(df, n=50)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_distributions(df)
    plot_correlation_matrix(df)
    plot_key_analysis(df)
    plot_mode_analysis(df)
    
    # Feature engineering
    df_encoded = engineer_features(df)
    
    # Prepare data for modeling
    X, y, feature_names = prepare_model_data(df_encoded)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    lr_model, y_pred_lr = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf_model, y_pred_rf = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
    
    # Plot confusion matrices
    plot_confusion_matrices(y_test, y_pred_lr, y_pred_rf)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - feature_distributions.png")
    print("  - correlation_matrix.png")
    print("  - key_analysis.png")
    print("  - mode_analysis.png")
    print("  - feature_importance.png")
    print("  - confusion_matrices.png")

if __name__ == "__main__":
    main()
