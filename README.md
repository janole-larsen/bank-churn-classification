# Churn Prediction: End-to-End Machine Learning Pipeline im Rahmen von CRISP DM

**Ziel:**
Vorhersage von Kundenabwanderung (Churn) zur datengetriebenen Optimierung von Kundenbindungsmaßnahmen in einer Retail-Bank.

**Prozessmodell:**
Struktur & Dokumentation orientieren sich am [CRISP-DM](https://www.ibm.com/docs/de/spss-modeler/saas?topic=dm-crisp-help-overview)-Framework.

---

## Inhaltsverzeichnis

1. [Business Understanding](#1-business-understanding)
2. [Data Understanding](#2-data-understanding)
3. [Data Preparation](#3-data-preparation)
4. [Modeling](#4-modeling)
5. [Evaluation](#5-evaluation)
6. [Deployment & Explainability](#6-deployment--explainability)
7. [Reproduzierbarkeit & Projektstruktur](#7-reproduzierbarkeit--projektstruktur)

---

## 1. Business Understanding

* **Use Case:**
  Ziel ist die Identifikation von Kunden mit erhöhtem Abwanderungsrisiko (`Exited`). Die Erhöhung der Bindungsquote ist wirtschaftlich relevant, da die Kosten für Kundenbindung (Retention Cost, RC) und Neukundengewinnung (Customer Acquisition Cost, CAC) im Entscheidungsprozess direkt einfließen.
* **Erfolgsmetriken:**
  Kostenoptimierte Churn-Vorhersage (Recall vs. Precision) mit Anpassung an Businessziele via Threshold/Kostenfunktion.

---

## 2. Data Understanding

* **Datensatz:**
  * 10.000 Kunden, 16 Variablen (demographisch, verhaltensbezogen, finanziell).
  * Original: https://www.openml.org/search?type=data&status=active&id=43390&sort=runs
  * Kaggle (Enriched): https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn
* **Explorative Analyse:**

  * *Profiling*: Verteilungen, Kreuztabellen, Zielklassen-Imbalance (80:20).
  * *Keine* fehlenden Werte, *keine* Duplikate.
  * *Erkenntnisse & Schlussfolgerungen für Preprocessing*:

| Variable               | Type      | Fehlend | Verteilung                                               | Relevanter linearer Zusammenhang Target              | Resultierendes Preprocessing                                        |
|------------------------|-----------|---------|----------------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------------|
| ORIGNAL DATASET        |           |         |                                                          |                                                      |                                                                     |
| `CustomerId`           | cat       | Nein    | -                                                        | -                                                    | Drop, nicht relevant                                                |
| `Surname`              | cat       | Nein    | Textvariable, viele Unique-Werte                         | nicht geprüft                                        | Drop, persönlicher Nachname                                         |
| `CreditScore`          | num       | Nein    | ~Normalverteilung, Peak bei 850                          | -                                                    | Skalieren (StandardScaler)                                          |
| `Geography`            | cat       | Nein    | Fra 50%, Ger 25%, Spa 25%                                | (Ger) 47% churn > (Fra, Spa) 20% churn               | One-Hot-Encoding                                                    |
| `Gender`               | cat       | Nein    | Ausgeglichen (≈ 50/50)                                   | (w) 32% churn > (m) 20%                              | One-Hot-Encoding                                                    |
| `Age`                  | num       | Nein    | ~Normalverteilung, leicht rechtsschief                   | corr = 0.38                                          | Skalieren (StandardScaler), ggf. Log/Power-Transform                |
| `Tenure`               | num       | Nein    | Gleichverteilt, aber Randhäufungen halbiert              | -                                                    | Skalieren (StandardScaler)                                          |
| `Balance`              | num       | Nein    | Bimodal punktmassig + ~Normalverteilung, 33 % exakt 0    | -                                                    | Skalieren (StandardScaler), neue Binärvariable `has_balance`        |
| `NumOfProducts`        | num       | Nein    | 1–4; 3/4 selten (3%)                                     | (3, 4): 82% churn > (1) 28% > (2) 1% churn           | (3+4) 3% minority → red flag behalten, rest Binarisierung           |
| `HasCrCard`            | cat       | Nein    | 30 % Ja, 70 % Nein, Imbalanced                           | -                                                    | Belassen → ja/nein                                                  |
| `IsActiveMember`       | cat       | Nein    | Gleichverteilung                                         | (False) 37% churn > (True) 15 % churn                | Belassen → ja/nein                                                  |
| `EstimatedSalary`      | num       | Nein    | Gleichverteilung                                         | -                                                    | Skalieren (StandardScaler)                                          |
| `Exited`               | cat       | Nein    | 20 % Ja, 80 % Nein                                       | x                                                    | Target → Imbalance lösen: class_weight oder sampling                |
| KAGGLE DATASET+        |           |         |                                                          |                                                      |                                                                     |
| `Complain`             | cat       | Nein    | 20 % Ja, 80 % Nein, Imbalanced                           | corr = 0.99 (!)                                      | Drop, da Target leakage                                             |
| `Satisfaction Score`   | num       | Nein    | Gleichverteilung                                         | unüblich (corr = 0)                                  | Drop, da resultiert aus Target leakage + unsinnige Corr             |
| `Card Type`            | cat       | Nein    | Gleichverteilung                                         | -                                                    | One-Hot-Encoding                                                    |
| `Point Earned`         | num       | Nein    | Mehrheitlich leicht schwankende Gleichverteilung         | -                                                    | Skalieren (StandardScaler)                                          |
| FEATURE ENGINEERING    |           |         |                                                          |                                                      |                                                                     |
| `has_balance`          | cat       | Nein    | 30 % Ja, 70 % Nein, Imbalanced                           | -                                                    | (Neues Feature!)                                                    |
| `multiproduct`         | cat       | Nein    | Ausgeglichen durch Aggregierung                          | -                                                    | (Neues Feature!)                                                    |
| `saving_rate`          | num       | Nein    | Resultierend durch Balance: Bimodal + Ausreißer          | -                                                    | (Neues Feature!) Rechts-Winsoring gegen Ausreißer                   |


---

## 3. Data Preparation

* **Feature Engineering:**

  * `no_balance`: Binär, ob Kontostand = 0
  * `multiproduct`: Binär, ≥2 Produkte
  * `multiproduct_redflag`: Binär, ≥3 Produkte für Churn Risiko Minority
  * `saving_rate`: Sparquote (Winsorized)
* **Feature Selection & Preprocessing:**

  * Drop: IDs, Surname, NumOfProducts (ersetzt durch `multiproduct`) und drop Target Leakage wie Complain.
  * Skalierung: Alle numerischen Variablen (`StandardScaler` und teils `Winsoring`)
  * Kategorisch: One-Hot-Encoding wie für `Geography`, `Gender`
* **Imbalance Handling:**

  * *Sampling*: None, Under, Over, SMOTE, BorderlineSMOTE (Vergleich & Visualisierung mit PCA, t-SNE, UMAP)
* **Automatisiertes Reporting:**
  Profiling vor/nach Preprocessing (`ydata_profiling`)



---

## 4. Modeling

* **Splitting:**
  Stratified: Train (70%) / Val (15%) / Test (15%), reproduzierbar via `random_state`.
* **Pipelines:**
  End-to-End via `sklearn` + `imblearn` (`feature_engineering` → Preprocessing → Sampling → Modell)
* **Modelle:**

  * Logistic Regression (LR)
  * Random Forest (RF)
  * XGBoost (XGB) <- Finales Modell
* **Hyperparameter-Tuning:**

  * `RandomizedSearchCV` + `Optuna
  * Zielmetrik: `average_precision` (threshholdunabhängiger, robust gg. Imbalance)
* **Sampling-Methoden:**
  Einbettung als Pipeline-Step → systematischer Vergleich.
* **Reproduzierbarkeit:**
  Zentrale `Config`-Klasse für alle Konstanten und Einstellungen.

---

## 5. Evaluation

* **Metriken:**

  * Precision, Recall, PR-AUC, Confusion Matrix
  * Kostenfunktion (abhängig von CAC/RC)
  * Threshold-Optimierung (minimale Gesamtkosten)
* **Kalibrierung:**

  * Isotonic-Kalibrierung
  * Brier Score
  * Kalibrierungskurve (Vorhersagewahrscheinlichkeit vs. tatsächlicher Anteil)
* **Generalisation:**

  * Lernkurve (Learning Curve)
  * Finales Generalisierungs-Gap (Train vs. Val)
* **Test-Set:**
  Finales Reporting (optimierter Schwellenwert, PR-Curve, Confusion Matrix)

| Set  | PR-AUC | Recall | Precision | Confusion Matrix   |
| ---- | ------ | ------ | --------- | ---------------------- |
| Val  | 0.68   | 0.80   | 0.42      | [[851, 343], [60, 246]] |
| Test | 0.69   | 0.71   | 0.58      | [[1038, 156], [89, 217]] |


\*Minimum, kostenoptimierter Schwellenwert (kalibriert) : 0.21 und 23k Euro Kosten

---

## 6. Deployment & Explainability

* **Explainability:**

  * Feature Importance (XGBoost)
  * SHAP: Global & Local (z.B. für Regulatorik, Transparenz, EU AI Act)
* **Stakeholder-Transparenz:**
  Modellentscheidungen nachvollziehbar, Dokumentation von Feature-Einfluss.
* **Next Steps:**

  * Optional: Deployment als API/Service
  * Monitoring, Modell-Update, Integration ins Kundensystem
  * Zusätzlich: Erkenntnisse während CRISP-DM wie Feature Einfluss Germany untersuchen -> warum genau gibt es hier hohe Churn? Wie verbessern?

---

## 7. Reproduzierbarkeit & Projektstruktur

**Pfadstruktur:**
```
DATA-MINING/
├── input/                          # Eingabedaten
│   └── data_churn_bank.csv         # Ursprünglicher Datensatz
├── output/                         # Automatisch erzeugte Reports (HTML)
│   ├── data_raw_report.html
│   └── data_preprocessed_report.html
├── churn_exploration.ipynb         # Explorative Analyse (EDA & Resampling)
├── churn_classification.ipynb      # Modellierung, Evaluation, Explainability
├── requirements.txt                # Python-Abhängigkeiten
└── .gitignore
```
**Zentrale Einstellungen:**
Alle Pfade, Parameter, Geschäftsgrößen und Modellparameter werden über die `Config`-Klasse gesteuert.

---

## 8. Quick Start

**Umgebung:**

```bash
git clone git@github.com:janole-larsen/bank-churn-classification.git
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Laufreihenfolge:**

1. EDA: `notebooks/churn_exploration.ipynb`
2. Modeling: `notebooks/churn_classification.ipynb`

---

## 9. Referenzen

* [CRISP-DM Process Model](https://www.ibm.com/docs/de/spss-modeler/saas?topic=dm-crisp-help-overview)
* Wichtige Bibliotheken: Scikit-learn, Imbalanced-learn, XGBoost, SHAP, optuna
* Die zugrunde liegende Literatur sowie Limitation, Einsatz KI findet sich in der begleitenden Hausarbeit wieder

---

*Alle zentralen Methoden & Entscheidungen sind zur Nachvollziehbarkeit in begleitender Ausarbeitung literaturbasiert weiter erläutert.*

---