# Hyperspectral Image Super-Resolution with Deep Transfer Learning

This repository contains my research project **Enhancing Hyperspectral Image Resolution with Deep Transfer Learning: A Case Study on Model Deployment and Evaluation**.  

We applied **Single-Image Hyperspectral Super-Resolution (SSPSR)** using a **Spatial-Spectral Prior Network (SSPN)**, trained on the **Chikusei dataset** (128 spectral bands, airborne scenes) and fine-tuned on the **CAVE dataset** (31 bands, lab indoor scenes).  

The project demonstrates both **data science expertise** (deep learning, transfer learning, statistical evaluation) and **data engineering practices** (efficient data preprocessing, reproducible pipelines, scalable model deployment).  

---

## 🔑 Highlights
  - Designed and trained deep learning models for hyperspectral image super-resolution.  
  - Introduced a **composite loss** (`L1 + TV + SAM + SSIM`) to balance fidelity, smoothness, spectral accuracy, and perceptual quality.  
  - Fine-tuned models on multiple datasets (Chikusei → CAVE), validating adaptability across domains.  
  - Achieved competitive **PSNR/SSIM** scores while reducing training cost.  
  - Built reproducible pipelines for preprocessing, patch extraction, and model training.  
  - Structured **50GB+ of high-dimensional spectral data** into efficient training/testing sets.  
  - Optimized model architecture (8-band variant) to cut training from **600s/epoch → 55s/epoch** (10× faster).  
  - Automated evaluation workflows with PQI metrics (PSNR, SSIM, ERGAS, SAM, RMSE).  

---

## 📊 Key Results
| Model | Dataset | Scale | MPSNR | MSSIM | ERGAS | SAM | Time/Epoch |
|-------|---------|-------|-------|-------|-------|-----|------------|
| SSPSR (Jiang, 2020) | Chikusei | ×4 | 40.36 | 0.941 | 4.98 | 2.35 | N/A |
| Full replication | Chikusei | ×4 | 40.33 | 0.944 | 4.95 | 2.38 | 600s |
| 8-band model | Chikusei | ×4 | 40.31 | 0.942 | 6.00 | 2.43 | **55s** |
| Fine-tuned model | CAVE | ×4 | 30.5 | 0.913 | — | — | — |

---

## 📂 Repository Structure

├─ cave_model/ # Fine-tuned weights for CAVE dataset
├─ trained_model/ # Trained model weights (ignored in .gitignore if large)
├─ mcodes/ # Utility modules
├─ EDA.ipynb # Exploratory data analysis
├─ Spectrum visualization.ipynb# Spectral visualization utilities
├─ Chikusei_demo.ipynb # Demo pipeline
├─ mains.py # Main training/evaluation script
├─ loss.py / loss_new.py # Loss functions (incl. composite loss)
├─ metrics.py # PQI metrics (PSNR, SSIM, ERGAS, SAM, RMSE)
├─ SSPSR.py # Model architecture (SSPN + SSB blocks)
├─ utils.py # Data preprocessing utilities
└─ demo.sh # Example run script


---

## ⚙️ Tech Stack
- **Languages/Frameworks:** Python, PyTorch, NumPy, Pandas  
- **Data Science Tools:** Jupyter Notebooks, Matplotlib, Scikit-learn  
- **Data Engineering Tools:** Custom preprocessing scripts, reproducible pipelines, `.gitignore` for datasets/models  
- **Datasets:** [Chikusei Hyperspectral](http://www.sal.t.u-tokyo.ac.jp/nus/dataset/Chikusei/) | [CAVE Multispectral](https://www.cs.columbia.edu/CAVE/databases/multispectral/)

---

## 📜 Citation
If you use this repo, please cite:  

**Nafis Khan, Siddharth Misra, James Omeke, Priyanka Chukka, Siddhanth Sirgapoor.**  
*Enhancing Hyperspectral Image Resolution with Deep Transfer Learning: A Case Study on Model Deployment and Evaluation*  

---

## 🚀 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train on Chikusei
python mains.py --dataset chikusei --scale 4 --epochs 40 --batch_size 32

# Fine-tune on CAVE
python mains.py --dataset cave --scale 4 --epochs 20 --lr 5e-5 --weights trained_model/chikusei_best.pt

