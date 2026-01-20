# Détection de Fraude sur les Compteurs Électriques et Gaz

## 📋 Description du Projet

Ce projet vise à détecter les clients frauduleux qui manipulent leurs compteurs électriques et de gaz pour réduire leurs factures. En utilisant l'apprentissage automatique, nous analysons l'historique de facturation pour identifier les patterns suspects et prédire la probabilité de fraude pour chaque client.

## 🎯 Objectif

Réduire les pertes financières d'une entreprise publique de distribution d'électricité et de gaz causées par des manipulations frauduleuses en détectant et identifiant les clients impliqués dans ces activités.

## 📊 Données

### Fichiers de Données

Les données sont disponibles dans les releases GitHub :

1. **Train (Entraînement)**
   - `client_train.csv` - Informations sur 135,493 clients
   - `invoice_train.csv` - 4,476,749 factures historiques

2. **Test**
   - `client_test.csv` - Informations sur 58,069 clients
   - `invoice_test.csv` - 1,939,730 factures

3. **Soumission**
   - `SampleSubmission.csv` - Format de soumission attendu

### Structure des Données

#### Données Client (`client_train.csv`)
- `client_id` : Identifiant unique du client
- `disrict` : District où se trouve le client
- `client_catg` : Catégorie du client
- `region` : Région du client
- `creation_date` : Date d'adhésion
- `target` : Fraude (1) ou Non-fraude (0)

#### Données Factures (`invoice_train.csv`)
- `client_id` : Identifiant du client
- `invoice_date` : Date de la facture
- `tarif_type` : Type de tarif
- `counter_number` : Numéro du compteur
- `counter_statue` : Statut du compteur
- `counter_code` : Code du compteur
- `reading_remarque` : Remarques de l'agent
- `counter_coefficient` : Coefficient de correction
- `consommation_level_1` à `consommation_level_4` : Niveaux de consommation
- `old_index` : Ancien indice
- `new_index` : Nouvel indice
- `months_number` : Nombre de mois
- `counter_type` : Type de compteur (ELEC/GAZ)

## 🔬 Méthodologie

### 1. Analyse Exploratoire (EDA)
- Exploration de la distribution des données
- Analyse du taux de fraude (~5.58% dans le dataset d'entraînement)
- Identification des patterns et anomalies

### 2. Ingénierie des Features

Les indicateurs de fraude développés incluent :

**Anomalies de Consommation:**
- Consommation totale (somme des 4 niveaux)
- Consommation par mois
- Ratio de consommation nulle
- Ratio de consommation très faible
- Volatilité de consommation (std/mean)

**Informations Compteur:**
- Nombre de compteurs uniques par client
- Statut moyen du compteur
- Remarques de lecture
- Coefficients de correction

**Patterns Temporels:**
- Durée de la relation client
- Fréquence des factures
- Âge du compte client

**Statistiques Agrégées:**
- Moyennes, écarts-types, min, max pour toutes les métriques
- Nombre de factures par client

### 3. Prétraitement
- Imputation des valeurs manquantes
- Encodage des variables catégorielles (LabelEncoder)
- Normalisation des features (StandardScaler)

### 4. Modélisation

**Modèle Choisi:** Gradient Boosting Classifier

**Paramètres:**
- `n_estimators`: 200 arbres
- `learning_rate`: 0.05
- `max_depth`: 5
- `min_samples_split`: 100
- `min_samples_leaf`: 50
- `subsample`: 0.8

**Justification:**
- Excellente performance sur données déséquilibrées
- Capture des relations non-linéaires complexes
- Robuste aux outliers
- Fournit l'importance des features

### 5. Performance

**Métriques sur l'ensemble de validation:**
- **ROC-AUC Score:** 0.8715
- **Accuracy:** 95%
- **Precision (Fraude):** 62%
- **Recall (Fraude):** 9%

**Top Features Importantes:**
1. `counter_number_nunique` (20.04%)
2. `counter_code_nunique` (16.21%)
3. `counter_statue_mean` (6.72%)
4. `client_duration_days` (6.08%)
5. `region` (5.44%)

## 🚀 Utilisation

### Prérequis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Téléchargement des Données

```bash
mkdir data
cd data
wget https://github.com/en2eavor/Data/releases/download/Data/client_train.csv
wget https://github.com/en2eavor/Data/releases/download/Data/invoice_train.csv
wget https://github.com/en2eavor/Data/releases/download/Data/client_test.csv
wget https://github.com/en2eavor/Data/releases/download/Data/invoice_test.csv
cd ..
```

### Exécution

**Option 1 : Script Python**
```bash
python fraud_detection.py
```

**Option 2 : Notebook Jupyter**
```bash
jupyter notebook fraud_detection_solution.ipynb
```

### Résultats

Le script génère :
- `SampleSubmission.csv` - Fichier de prédictions avec probabilités de fraude

Format du fichier de soumission :
```csv
client_id,target
test_Client_0,0.02275204255953885
test_Client_1,0.1375587613433723
...
```

## 📈 Résultats

### Statistiques des Prédictions

- **Total de clients testés:** 58,069
- **Clients suspects (prob > 0.5):** 482
- **Proportion de fraude prédite:** 0.83%

### Distribution des Probabilités

La majorité des clients ont une faible probabilité de fraude, ce qui est cohérent avec le taux de fraude observé dans les données d'entraînement.

## 💡 Recommandations pour l'Entreprise

### Priorisation des Inspections
1. **Haute Priorité** (prob > 0.7) : Inspection immédiate
2. **Moyenne Priorité** (0.5 < prob < 0.7) : Investigation approfondie
3. **Surveillance** (0.3 < prob < 0.5) : Monitoring renforcé

### Actions Préventives
- Installation de compteurs intelligents
- Audits réguliers des clients à risque
- Système de détection en temps réel
- Sensibilisation sur les conséquences légales

### Amélioration Continue
- Feedback loop avec résultats des inspections
- Mise à jour régulière du modèle
- Intégration de nouvelles sources de données

## 🔍 Indicateurs de Fraude Détectés

Le modèle identifie les patterns suivants comme suspects :

1. **Changements fréquents de compteur** - Indicateur le plus fort
2. **Codes de compteur multiples** - Manipulations possibles
3. **Statut de compteur anormal** - Dysfonctionnements suspects
4. **Consommation volatile** - Variations inhabituelles
5. **Consommation très faible** - Sous-déclaration potentielle

## 📝 Structure du Projet

```
.
├── README.md                          # Documentation
├── fraud_detection.py                 # Script principal
├── fraud_detection_solution.ipynb     # Notebook Jupyter détaillé
├── SampleSubmission.csv               # Prédictions finales
├── Data/
│   └── SampleSubmission.csv           # Exemple de format
└── data/                              # Données (à télécharger)
    ├── client_train.csv
    ├── invoice_train.csv
    ├── client_test.csv
    └── invoice_test.csv
```

## 🛠️ Technologies Utilisées

- **Python 3.x**
- **pandas** - Manipulation de données
- **numpy** - Calculs numériques
- **scikit-learn** - Modélisation ML
- **matplotlib/seaborn** - Visualisation

## 📊 Améliorations Futures

1. **Modèles Avancés**
   - XGBoost, LightGBM
   - Réseaux de neurones (Deep Learning)
   - Modèles d'ensemble

2. **Features Additionnelles**
   - Données géospatiales
   - Patterns saisonniers
   - Données météorologiques
   - Profils de consommation horaires

3. **Techniques d'Équilibrage**
   - SMOTE pour le déséquilibre de classes
   - Ajustement des poids de classes
   - Sous-échantillonnage stratégique

4. **Déploiement**
   - API REST pour prédictions en temps réel
   - Dashboard de monitoring
   - Système d'alertes automatisées

## 👥 Auteur

Solution développée pour la détection de fraude dans les compteurs électriques et de gaz.

## 📄 Licence

Ce projet est développé à des fins éducatives et professionnelles.

---

**Note:** Les prédictions sont des probabilités entre 0 et 1. Un seuil de 0.5 est utilisé par défaut, mais peut être ajusté selon la tolérance au risque de l'entreprise.
