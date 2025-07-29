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
  10.000 Kunden, 13 Variablen (demographisch, verhaltensbezogen, finanziell).
* **Explorative Analyse:**

  * *Profiling*: Verteilungen, Kreuztabellen, Zielklassen-Imbalance (80:20).
  * *Keine* fehlenden Werte, *keine* Duplikate.
  * *Erkenntnisse & Schlussfolgerungen für Preprocessing*:
| Feature         | Wichtigkeit für Modell | Verteilung/Besonderheit                  | Empfehlung/Fazit                               |
| --------------- | ---------------------- | ---------------------------------------- | ---------------------------------------------- |
| CreditScore     | Hoch                   | \~Normal, Peak bei 850                   | Stark erklärend für Churn, skalieren           |
| Geography       | Mittel                 | Gleichmäßig verteilt                     | Einfluss regional, One-Hot-Encoding            |
| Gender          | Niedrig                | \~50/50                                  | Kein Bias nachweisbar, One-Hot-Encoding        |
| Age             | Hoch                   | Leicht schief, jüngere kündigen seltener | Transformieren & skalieren, starke Korrelation |
| Tenure          | Mittel                 | Gleichmäßig, wenig Einfluss              | Skalieren, geringer Impact                     |
| Balance         | Hoch                   | 33 % exakt 0, Rest normal                | Bimodalität, Feature „has\_balance” hilft      |
| NumOfProducts   | Hoch                   | Unausgeglichen, meist 1 Produkt          | Neue Variable „multiproduct“ nutzen            |
| HasCrCard       | Gering                 | 30 % Ja, 70 % Nein                       | Geringer Impact                                |
| IsActiveMember  | Hoch                   | Gleichmäßig                              | Wichtig für Churn, als binär belassen          |
| EstimatedSalary | Niedrig                | Gleichverteilung                         | Kein signifikanter Zusammenhang                |
| Exited (Target) | —                      | 20 % Ja, 80 % Nein                       | **Target:** Starke Imbalance, beachten!        |
| has\_balance    | Mittel                 | 30/70 Imbalance                          | Nützliches Feature (neu)                       |
| multiproduct    | Mittel                 | Nach Binarisierung ausgeglichen          | Nützliches Feature (neu)                       |
| saving\_rate    | Niedrig                | Bimodal, Ausreißer                       | Nützliches Feature (neu) winsorieren           |


* **Tools:**
  `pandas`, `ydata_profiling`, `seaborn`

---

## 3. Data Preparation

* **Feature Engineering:**

  * `no_balance`: Binär, ob Kontostand = 0
  * `multiproduct`: Binär, ≥2 Produkte
  * `saving_rate`: Sparquote (Winsorized)
* **Feature Selection & Preprocessing:**

  * Drop: IDs, Surname, NumOfProducts (ersetzt durch `multiproduct`)
  * Skalierung: Alle numerischen Variablen (`StandardScaler`)
  * Kategorisch: One-Hot-Encoding für `Geography`, `Gender`
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
  * XGBoost (XGB)
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

| Set  | PR-AUC | Recall | Precision | Kosten\*   |
| ---- | ------ | ------ | --------- | ---------- |
| Val  | 0.60   | 0.62   | 0.50      | \~22.700 € |
| Test | 0.52   | 0.57   | 0.46      | -          |

\*Minimum, kostenoptimierter Schwellenwert (kalibriert)

---

## 6. Deployment & Explainability

* **Explainability:**

  * Feature Importance (XGBoost)
  * SHAP: Global & Local (z.B. für Regulatorik, EU AI Act)
* **Stakeholder-Transparenz:**
  Modellentscheidungen nachvollziehbar, Dokumentation von Feature-Einfluss.
* **Next Steps:**

  * Optional: Deployment als API/Service
  * Monitoring, Modell-Update, Integration ins Kundensystem

---

## 7. Reproduzierbarkeit & Projektstruktur

**Pfadstruktur:**

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

1. EDA: `notebooks/1_EDA.ipynb`
2. Modeling: `notebooks/2_Modeling.ipynb`

---

## 9. Referenzen

* [CRISP-DM Process Model](https://www.ibm.com/docs/de/spss-modeler/saas?topic=dm-crisp-help-overview)
* Wichtige Bibliotheken: Scikit-learn, Imbalanced-learn, XGBoost, SHAP, optuna
* Die zugrunde liegende Literatur sowie Limitation, Einsatz KI findet sich in der begleitenden Hausarbeit wieder

---

*Alle zentralen Methoden & Entscheidungen sind zur Nachvollziehbarkeit in begleitender Ausarbeitung literaturbasiert umfassender erläutert.*

---



