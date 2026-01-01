# Analyzing ECG Data for Arrythmia Detection 
### Comparative Analysis: Feature Engineering + ML vs. Deep Learning

![Status](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Tech](https://img.shields.io/badge/TensorFlow-Scikit--Learn-orange)
![Domain](https://img.shields.io/badge/Domain-Biomedical_Engineering-red)

## Summary
This project challenges the industry trend of relying solely on deep learning for biological signal processing. It benchmarks a 1D-Convolutional Neural Network (CNN) against a lightweight Logistic Regression classifier that utilizes domain-specific feature engineering, namely in nonlinear dynamics. 

**Hypothesis:** A model grounded in physiological principles can achieve comparable diagnostic performance to a "black box" neural network while offering superior interpretability and magntitudes of reduction in training time. 


## Background and Motivation



### Biological Context
The heart is not just a muscle; it is an electromechanical pump controlled by a complex biological circuit.
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
This project hypothesizes that we do not need a neural network to learn these patterns from scratch. We can explicitly engineer features that map directly to the clinician's checklist:

| Clinical Feature | Domain | Feature Engineering |
| :--- | :--- | :--- |
| **Rhythm Stability** | **Non-Linear Dynamics** | **Poincaré Plots:** We map each beat interval ($t_n$) against the next ($t_{n+1}$). A stable heart creates a tight cluster; a "chaotic" heart creates a scattered cloud. We quantify this with **Entropy** and **SD1/SD2** geometry. |
| **Signal Shape** | **Statistical Moments** | **Kurtosis:** A healthy beat is a sharp spike (i.e high kurtosis). A blocked beat (e.g. LBBB) is a wide, sluggish wave (i.e. low kurtosis). |
| **Signal Direction** | **Distribution Asymmetry** | **Skewness:** A Premature Ventricular Contraction (PVC) originates from the bottom of the heart, reversing the signal polarity - this flips the statistical skew of the wave. |

The utilization of feature engineering not only enables construction of a lightweight ML model, but also allows for extremely interpretable insights into predicted cases of arrythmia. 

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
    * `extract_features`: Generates 11 domain features including SD1/SD2, Kurtosis/Skewness, and Sample Entropy.
4.  **Model Definitions:**
    * **Engineering Pipeline:** StandardScaler -> Logistic Regression.
    * **Deep Learning Pipeline:** 2-layer 1D-CNN with Batch Normalization and Dropout.
5.  **Visualization Functions:**
    * `plot_chaos_gallery`: Visualizes Poincaré plots for different arrhythmia classes.
    * `visualize_interpretability`: Compares CNN Saliency Maps vs. Engineering Feature Space.
<img width="1776" height="611" alt="image" src="https://github.com/user-attachments/assets/c4d7a84d-97ac-426e-9c53-8f5deafbef59" />

## Results and Evaluations
The comparative performance was measured in two different tests: a simple binary test and a more challenging multiclass test. 
The study resulted in a comparison between the two approaches across 5 classes (Normal, LBBB, RBBB, PVC, APC).

| Metric | Engineering Model (Logistic Reg) | Deep Learning (1D-CNN) | Winner |
| :--- | :--- | :--- | :--- |
| **Training Time** | **~0.02 Seconds** | ~30.0 Seconds | **Engineering (1500x Faster)** |
| **Interpretability** | **Transparent (Physics-based)** | Opaque (Saliency Maps) | **Engineering** |
| **RBBB Recall** | **0.90** | 0.82 | **Engineering** |
| **PVC Recall** | 0.49 | **0.54** | Deep Learning |
| **Overall Accuracy** | ~83% | ~86% | Deep Learning (Marginal) |



<img width="1989" height="1189" alt="image" src="https://github.com/user-attachments/assets/487cbf4c-3d7d-47f3-a0aa-e212b424b9b0" />


<img width="1570" height="702" alt="image" src="https://github.com/user-attachments/assets/b9d66f3a-af0b-4d47-9685-b57df35cbd10" />





**Key Findings:**
* **Efficiency:** The Engineering model is lightweight enough to run on ultra-low-power edge devices (e.g., smartwatches) without GPU acceleration.

## Future Work
* **Edge Deployment:** Port the HeartEngineer logic to C++ for embedded microcontroller testing.
* **Cross-Dataset Validation:** Test robustness on the AHA or PTB Diagnostic ECG Database.
* **Hybrid Architecture:** Implement a stack where the Engineering model handles initial screening (for speed) and the CNN handles ambiguous cases (for precision).
