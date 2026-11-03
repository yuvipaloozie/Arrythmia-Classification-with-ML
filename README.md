# Analyzing ECG Data for Arrythmia Detection 
### Comparative Analysis: Feature Engineering + ML vs. Deep Learning
  ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-F7931E?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![wfdb](https://img.shields.io/badge/wfdb-2563A6?style=for-the-badge)

  
## Summary
This project compares a 1D-Convolutional Neural Network (CNN) with logistic regression on engineered ECG features. It focuses on the tradeoffs between classification performance, training time and interpretation, using nonlinear-dynamics and signal-shape features. 

**Hypothesis:** Physiologically motivated features can give a small classifier comparable ECG-window classification performance, shorter training time and features that are easier to inspect. 

## Background and Motivation

### Biological Context
The heart is an electromechanical pump whose contractions depend on electrical conduction.
* **The Signal:** Every heartbeat is triggered by an electrical impulse from the Sinoatrial (SA) Node. This impulse travels down conductive pathways (His-Purkinje system), causing the muscle fibers to contract.
* **The ECG:** An Electrocardiogram (ECG) measures the voltage changes on the skin caused by this electrical wave. A normal heartbeat produces a specific shape called the **P-QRS-T complex**:
    * **P-wave:** Atrial contraction.
    * **QRS Complex:** Ventricular contraction (the main "spike").
    * **T-wave:** Resetting (repolarization).

### Defining Arrythmia
An arrhythmia is any deviation from the normal rate or rhythm of the heart. Clinicians generally look for two distinct types of failures:
1.  **Failures of Rhythm:** There are abnormalities in the frequency of the expected signal as the heart is "misfiring". This can manifest as irregular R-R intervals. 
2.  **Failures of Conduction:** The specific path for the electrical signal is altered due to blockage. This can manifest as a change in the shape of the QRS complex. 

### Feature Engineering
The feature-engineering branch encodes rhythm and signal-shape information explicitly. These features provide interpretable measurements to compare with the CNN:

| Clinical Feature | Domain | Feature Engineering |
| :--- | :--- | :--- |
| **Rhythm Stability** | **Non-Linear Dynamics** | **Poincaré Plots:** We map each beat interval ($t_n$) against the next ($t_{n+1}$). The spread of successive R-R intervals is summarized by **SD1/SD2** geometry. The feature set also includes **entropy** of the filtered waveform's amplitude histogram. |
| **Signal Shape** | **Statistical Moments** | **Kurtosis:** The notebook uses kurtosis to summarize the filtered waveform's amplitude distribution and compare signal-shape differences between classes. |
| **Signal Direction** | **Distribution Asymmetry** | **Skewness:** The notebook uses skewness to summarize waveform asymmetry. Its relationship to the class label is assessed in the classification comparison. |

These engineered features make it possible to inspect which rhythm and signal-shape measurements contribute to an arrhythmia classification. 

<img width="1490" height="1151" alt="image" src="https://github.com/user-attachments/assets/a49d9ed8-5987-4c2a-b663-8a04ae3ebe32" />

## Data Source and Processing
* **Source:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (PhysioNet).
* **Specifications:** 48 half-hour excerpts of 2-channel ECG recordings at 360 samples per second.
    * Beats in each plot are annotated using a letter code (e.g. 'N' - normal, 'V' - Premature ventricular contraction, 'A' - Atrial premature beat, etc.)
    * Further info for each of the records (patient profile, medication, etc.) can be found at link
* **Preprocessing Pipeline:**
    * **Noise Removal:** 5-15Hz Bandpass filter implementation to remove noisy signals. 
    * **Peak Detection:** Robust algorithm created from scratch to adjust for baseline wander. 
    * **Windowing:** Signals sliced into 10-second non-overlapping windows.
    * **Labeling Strategy:** Windows are classified based on the most "severe" beat present (Hierarchy: Ventricular > Atrial > Conduction Block > Normal).

## Major Libraries
* **Signal Processing:** `wfdb`, `scipy` 
* **Machine Learning:** `scikit-learn` 
* **Deep Learning:** `tensorflow` / `keras`
* **Data Manipulation:** `numpy`, `pandas`
* **Visualization:** `matplotlib`, `seaborn`

## Code Structure
The project is contained within a single reproducible notebook (`arrythmiaml.ipynb`) designed to narrate the comparison and run end-to-end. 
**Note:** The comparison between models was repeated for two different tasks - binary classification (normal vs _any_ arrythmia) and multiclass classification (Normal vs LBBB vs RBBB etc.)

1.  **Config Class:** Centralized configuration for sample rates, window sizes, and paths.
2.  **download_full_dataset():** Data ingestion directly from PhysioNet.
3.  **HeartEngineer Class:**
    * `pan_tompkins_detector`: Robust R-peak detection.
    * `extract_features`: Generates 11 domain features including SD1/SD2, kurtosis/skewdness, and histogram-based Shannon entropy.
4.  **Model Definitions:**
    * **Engineering Pipeline:** StandardScaler -> Logistic Regression.
    * **Deep Learning Pipeline:** 2-layer 1D-CNN with Batch Normalization and Dropout. Model was trained 20-30 epochs for efficiency. 
5.  **Visualization Functions:**
    * `plot_chaos_gallery`: Visualizes Poincaré plots for different arrhythmia classes.
    * `visualize_interpretability`: Compares CNN saliency maps vs. engineering feature space.
<img width="1776" height="611" alt="image" src="https://github.com/user-attachments/assets/c4d7a84d-97ac-426e-9c53-8f5deafbef59" />

## Results and Evaluations
The comparative performance was measured in two different tests: a simple binary test and a more challenging multiclass test (Normal, LBBB, RBBB, PVC, APC). These are window-level train/test results; the notebook does not implement a patient-held-out evaluation.

### Model Performance

| Metric / Disease Class | Logistic Reg | 1-D CNN |
| :--- | :--- | :--- |
| **Training Time** | **0.03s** | 37.38s |
| **Binary Accuracy** | **83.0%** | 82.8% |
| **Binary Recall** | 76.1% | **84.2%** |
| **Binary Precision** | **74.6%** | 70.9% |
| **Binary F1-Score** | 75.3% | **77.0%** |
| **Recall: Normal Rhythm** | **53%** | 42% |
| **Recall: LBBB (Left Block)** | 75% | **82%** |
| **Recall: RBBB (Right Block)**| **90%** | 82% |
| **Recall: PVC (Ventricular)** | 49% | **54%** |
| **Recall: APC (Atrial)** | 46% | **56%** |

**Binary Classification**
<img width="1141" height="336" alt="image" src="https://github.com/user-attachments/assets/cff1486f-0497-49bf-9743-4994ee46bb84" />

**Multi-class Classification**
<img width="1570" height="702" alt="image" src="https://github.com/user-attachments/assets/3a39f6cc-7e93-4a3f-8874-61b01a4449a0" />

### Interpretability

Interpretability is part of the comparison. Logistic regression operates on named ECG features, so its decisions can be examined in a compact feature space. The CNN uses the waveform directly; gradient-based saliency provides a different view of which signal regions influence its predictions. The plots below compare those views for this dataset. 

**Logistic Regression + Engineering**
* Differences in diagnoses were driven by lower-dimensionality representation in Poincare plots, namely SD1-SD2 values
* Clear separation when doing broad binary classification, but harder to distinguish between specific classes using SD1-SD2 values/other statistical moments in some cases (e.g. Normal vs LBBB)
<img width="1467" height="706" alt="image" src="https://github.com/user-attachments/assets/febece05-a9d2-432e-b003-96bc106e441a" />
<img width="1141" height="675" alt="image" src="https://github.com/user-attachments/assets/798a9b75-a653-49af-a3e5-25ebbdc956f6" />

**1-D CNN**
* Input-gradient saliency estimates which waveform regions affect the prediction
* CNN weighs more attention on points closer to QRS complex/R-R intervals
<img width="1175" height="626" alt="image" src="https://github.com/user-attachments/assets/deed633a-65ad-4e08-9b8f-5485d4a3e820" />

### Key Findings
* In the recorded binary run, logistic regression reaches 83.0% accuracy and the CNN reaches 82.8%. Their reported training times are 0.03s and 37.38s; the CNN has higher binary recall and F1.
* The multiclass results vary by class. Logistic regression has higher RBBB recall, while the CNN has higher recall for LBBB, PVC and APC. The comparison separates those class-level tradeoffs from overall accuracy.

## Future Work
* **Edge Deployment:** Port the HeartEngineer logic to C++ for embedded microcontroller testing.
* **Cross-Dataset Validation:** Test robustness on other ECG databases that are available.
* **Hybrid Classifier:** Utilize a dual-model approach with LogReg as a first pass, then a finely tuned CNN for edge cases.

