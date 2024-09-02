# A-Comparative-Study-of-Detection-of-Tuberculosis-using-Machine-Learning-Deep-Learning


## Overview

This Jupyter Notebook, `tb.ipynb`, was developed as part of the research paper titled *A Comparative Study of Detection of Tuberculosis using Machine Learning & Deep Learning*. The paper was published in the proceedings of the 2023 10th International Conference on Computing for Sustainable Global Development (INDIACom) by IEEE. This notebook accompanies the paper and provides practical insights into the comparative analysis conducted on various machine learning (ML) and deep learning (DL) models for the detection of tuberculosis (TB).

## Purpose

The purpose of this notebook is to demonstrate the implementation and comparison of different ML and DL models used in TB diagnosis, including:

- **Machine Learning Models**: Algorithms and techniques traditionally used in TB diagnosis.
- **Deep Learning Models**: Advanced neural network models, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), along with hybrid models, to analyze TB datasets.

The notebook provides code, analysis, and results that align with the methodologies discussed in the paper.

## Dataset, Model, and Techniques

After experimenting with various models and preprocessing methods, we finalized the following setup for the TB detection task:

- **Dataset**: TBX11K
- **Model**: EfficientNetB3, leveraging transfer learning for enhanced performance in TB detection.
- **Pre-processing**: 
  - **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Used to enhance the contrast of the images.
  - **ImageDataGenerator**: Applied for data augmentation to increase the diversity of the training data.
- **Visualization**: 
  - **GradCAM** (Gradient-weighted Class Activation Mapping): Utilized to visualize the areas of the image that the model focuses on during the classification process.

## Structure

The notebook is organized into several sections:

1. **Introduction**: Brief overview and background on TB and the motivation behind using ML and DL for its detection.
2. **Data Preparation**: Steps involved in preparing and preprocessing TB datasets, including chest X-ray images, sputum smear microscopy, and genomics data.
3. **Model Implementation**: Implementation of various ML and DL models. This section contains the code for training, testing, and evaluating the models.
4. **Comparative Analysis**: Evaluation metrics and comparison of the models based on accuracy, precision, recall, and other relevant metrics.
5. **Results & Discussion**: Insights and observations drawn from the comparative study.
6. **Conclusion**: Summary of findings and their implications for TB diagnosis.
7. **Future Scope**: Potential areas for further research and improvement in the application of DL in TB diagnosis.

## Prerequisites

- **Python 3.x**: Ensure you have Python installed on your machine.
- **Jupyter Notebook**: You can use Jupyter Notebook or JupyterLab to run this `.ipynb` file.
- **Required Libraries**: The following Python libraries are used in this notebook:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tensorflow` or `keras`
  - `matplotlib`
  - `seaborn`

Install the required libraries using `pip`:

```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn
