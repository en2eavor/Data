#!/usr/bin/env python3
"""
Détection de Fraude sur les Compteurs Électriques et Gaz

Script de solution complète pour détecter les clients frauduleux
en analysant l'historique de facturation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_invoice_features(invoice_df):
    """
    Créer des features agrégées à partir des données de facturation
    
    Cette fonction génère des indicateurs de fraude potentiels:
    - Anomalies de consommation (très faible, nulle, volatile)
    - Problèmes de compteur (statut, remarques)
    - Patterns temporels (fréquence des factures)
    """
    df = invoice_df.copy()
    
    # Conversion des types
    df['counter_statue'] = pd.to_numeric(df['counter_statue'], errors='coerce')
    
    # Conversion de la date
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    
    # Calcul de la consommation totale
    df['total_consumption'] = (df['consommation_level_1'] + 
                               df['consommation_level_2'] + 
                               df['consommation_level_3'] + 
                               df['consommation_level_4'])
    
    # Différence d'index
    df['index_diff'] = df['new_index'] - df['old_index']
    
    # Consommation par mois
    df['consumption_per_month'] = df['total_consumption'] / (df['months_number'] + 0.01)
    df['index_diff_per_month'] = df['index_diff'] / (df['months_number'] + 0.01)
    
    # Features agrégées par client
    agg_features = df.groupby('client_id').agg({
        # Statistiques de consommation
        'total_consumption': ['mean', 'std', 'min', 'max', 'sum'],
        'consumption_per_month': ['mean', 'std', 'min', 'max'],
        'index_diff': ['mean', 'std', 'min', 'max'],
        'index_diff_per_month': ['mean', 'std'],
        
        # Statistiques par niveau de consommation
        'consommation_level_1': ['mean', 'sum'],
        'consommation_level_2': ['mean', 'sum'],
        'consommation_level_3': ['mean', 'sum'],
        'consommation_level_4': ['mean', 'sum'],
        
        # Informations sur le compteur
        'counter_statue': ['mean', 'std', 'nunique'],
        'counter_coefficient': ['mean', 'std', 'max'],
        'reading_remarque': ['mean', 'std', 'max'],
        'counter_code': ['nunique'],
        'counter_number': ['nunique'],
        
        # Informations temporelles
        'months_number': ['mean', 'sum', 'max'],
        'invoice_date': ['count', 'min', 'max']
    })
    
    # Aplatir les noms de colonnes
    agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns.values]
    agg_features.reset_index(inplace=True)
    
    # Features additionnelles
    agg_features['invoice_count'] = agg_features['invoice_date_count']
    
    # Durée de la relation client (en jours)
    agg_features['client_duration_days'] = (
        agg_features['invoice_date_max'] - agg_features['invoice_date_min']
    ).dt.days
    
    # Fréquence moyenne des factures
    agg_features['avg_invoice_frequency'] = (
        agg_features['client_duration_days'] / (agg_features['invoice_count'] + 0.01)
    )
    
    # Ratio de consommation nulle (indicateur de fraude)
    zero_consumption = df.groupby('client_id')['total_consumption'].apply(
        lambda x: (x == 0).sum() / len(x)
    ).reset_index()
    zero_consumption.columns = ['client_id', 'zero_consumption_ratio']
    agg_features = agg_features.merge(zero_consumption, on='client_id', how='left')
    
    # Ratio de consommation très faible
    low_consumption = df.groupby('client_id')['total_consumption'].apply(
        lambda x: (x < x.quantile(0.1)).sum() / len(x)
    ).reset_index()
    low_consumption.columns = ['client_id', 'low_consumption_ratio']
    agg_features = agg_features.merge(low_consumption, on='client_id', how='left')
    
    # Volatilité de la consommation
    agg_features['consumption_volatility'] = (
        agg_features['total_consumption_std'] / (agg_features['total_consumption_mean'] + 0.01)
    )
    
    # Type de compteur majoritaire
    counter_type_mode = df.groupby('client_id')['counter_type'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNKNOWN'
    )
    counter_type_mode = counter_type_mode.reset_index()
    counter_type_mode.columns = ['client_id', 'counter_type_mode']
    agg_features = agg_features.merge(counter_type_mode, on='client_id', how='left')
    
    # Supprimer les colonnes de date
    agg_features = agg_features.drop(['invoice_date_min', 'invoice_date_max'], axis=1)
    
    return agg_features


def prepare_client_features(client_df):
    """
    Préparer les features client
    """
    df = client_df.copy()
    
    # Conversion de la date de création
    df['creation_date'] = pd.to_datetime(df['creation_date'], format='%d/%m/%Y')
    
    # Features temporelles
    df['creation_year'] = df['creation_date'].dt.year
    df['creation_month'] = df['creation_date'].dt.month
    df['client_age_years'] = (datetime.now() - df['creation_date']).dt.days / 365.25
    
    # Supprimer la colonne date originale
    df = df.drop('creation_date', axis=1)
    
    return df


def main():
    """
    Fonction principale pour exécuter la détection de fraude
    """
    print("="*80)
    print("DÉTECTION DE FRAUDE - COMPTEURS ÉLECTRIQUES ET GAZ")
    print("="*80)
    
    # 1. CHARGEMENT DES DONNÉES
    print("\n[1/7] Chargement des données...")
    client_train = pd.read_csv('data/client_train.csv')
    invoice_train = pd.read_csv('data/invoice_train.csv')
    client_test = pd.read_csv('data/client_test.csv')
    invoice_test = pd.read_csv('data/invoice_test.csv')
    
    print(f"  - Client Train: {client_train.shape}")
    print(f"  - Invoice Train: {invoice_train.shape}")
    print(f"  - Client Test: {client_test.shape}")
    print(f"  - Invoice Test: {invoice_test.shape}")
    print(f"  - Taux de fraude: {client_train['target'].mean():.2%}")
    
    # 2. INGÉNIERIE DES FEATURES
    print("\n[2/7] Ingénierie des features...")
    train_invoice_features = create_invoice_features(invoice_train)
    test_invoice_features = create_invoice_features(invoice_test)
    print(f"  - Features factures créées: {train_invoice_features.shape[1] - 1}")
    
    # 3. PRÉPARATION DES DONNÉES CLIENT
    print("\n[3/7] Préparation des données client...")
    client_train_prep = prepare_client_features(client_train)
    client_test_prep = prepare_client_features(client_test)
    
    # 4. FUSION ET PRÉPARATION FINALE
    print("\n[4/7] Fusion des features...")
    X_train = client_train_prep.merge(train_invoice_features, on='client_id', how='left')
    X_test = client_test_prep.merge(test_invoice_features, on='client_id', how='left')
    
    # Sauvegarder les IDs de test
    test_ids = X_test['client_id'].copy()
    
    # Extraire la cible
    y_train = X_train['target'].copy()
    X_train = X_train.drop(['target', 'client_id'], axis=1)
    X_test = X_test.drop(['client_id'], axis=1)
    
    print(f"  - Features totales: {X_train.shape[1]}")
    
    # Prétraitement
    print("\n[5/7] Prétraitement...")
    
    # Encodage des variables catégorielles d'abord
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    print(f"  - Encodage de {len(categorical_cols)} colonnes catégorielles...")
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Gérer les NaN avant l'encodage
        X_train[col] = X_train[col].fillna('UNKNOWN')
        X_test[col] = X_test[col].fillna('UNKNOWN')
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
    
    # Gestion des valeurs manquantes pour les colonnes numériques
    print("  - Imputation des valeurs manquantes...")
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
    X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].median())
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. ENTRAÎNEMENT DU MODÈLE
    print("\n[6/7] Entraînement du modèle Gradient Boosting...")
    
    # Split pour validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Entraînement initial pour validation
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=100,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_tr, y_tr)
    
    # Évaluation
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    print("\n  === PERFORMANCE SUR VALIDATION ===")
    print(classification_report(y_val, y_val_pred, target_names=['Non-Fraude', 'Fraude']))
    print(f"  ROC-AUC Score: {roc_auc_score(y_val, y_val_proba):.4f}")
    
    # Entraînement final sur toutes les données
    print("\n  Réentraînement sur toutes les données...")
    final_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=100,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    final_model.fit(X_train_scaled, y_train)
    
    # 6. PRÉDICTIONS
    print("\n[7/7] Génération des prédictions sur le test set...")
    test_predictions_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"  - Prédictions générées: {len(test_predictions_proba)}")
    print(f"  - Clients suspects (prob > 0.5): {(test_predictions_proba > 0.5).sum()}")
    print(f"  - Proportion: {(test_predictions_proba > 0.5).mean():.2%}")
    
    # 7. SAUVEGARDE
    submission = pd.DataFrame({
        'client_id': test_ids,
        'target': test_predictions_proba
    })
    
    submission.to_csv('SampleSubmission.csv', index=False)
    print("\n✓ Fichier de soumission sauvegardé: SampleSubmission.csv")
    
    # Top 10 des features importantes
    print("\n=== TOP 10 FEATURES LES PLUS IMPORTANTES ===")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "="*80)
    print("TRAITEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*80)


if __name__ == "__main__":
    main()
