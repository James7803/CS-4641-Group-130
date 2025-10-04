# Brain Tumor Classification Project

# Introduction

Brain tumors are abnormal growths within the brain. Tumors cause pressure build-up within the skull, often leading to severe complications and death. Cancerous or not, tumors interfere with critical neurological functions, necessitating timely detection.  

Earlier literature demonstrates that traditional image processing often struggles with categorizing brain tumors due to their wide variability. However, recent developments in deep learning have shown promise in improving accuracy in tumor classification.  

To amass data about tumors, we are using the publicly available [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). This dataset contains over seven thousand Magnetic Resonance Imaging (MRI) scans of human brains. Each image is labeled by its respective tumor category or lack thereof.  

# Problem Definition

Diagnosing brain tumors from medical images is a complex process that is not always foolproof. Our project will give a second opinion for radiologists to ensure that they have made little to no oversights in their diagnosis of the patient into four distinct categories: Glioma, meningioma (non-cancerous), pituitary, and no tumor (healthy). Additionally, classifying the tumor by grade will aid in streamlining patients’ treatment. 

Diagnosing tumors is further complicated by variability in MRI acquisition and tumor appearance, which can lead to inconsistent readings across clinicians. Our system aims to provide a consistent, reproducible second opinion across the four classes, as well as offer a preliminary tumor-grade suggestion to help triage cases and streamline follow-up imaging and treatment planning. 


# Methodology

To classify brain tumors from MRI scans, our team’s approach combines robust preprocessing, supervised deeper learning models, and transfer learning to maximize accuracy and generalizability.  

## Data Processing

All images will be normalized so that pixel intensities fall within a standard range between 0 and 1, which stabilizes gradient descent during training. Images will then be resized to a consistent resolution like `224x224` to match the input requirements of common convolutional neural network (CNN) architectures.  

Data segmentation methods such as random rotations, horizontal/vertical flips, zoom, and contrast adjustments will be applied using:  

- `tensorflow.keras.preprocessing.image.ImageDataGenerator`  
- `torchvision.transforms`  

to artificially expand the dataset and reduce overfitting.  

As a final processing measure, the team will perform train-validation-test splits with stratification to ensure balanced class distribution across the four tumor categories.  

## Machine Learning Models

The team determined that convolutional neural networks (CNNs) would serve as great primary models for this application, given that they are great fits for spatial image recognition features.  

A baseline CNN will be constructed with multiple convolution, pooling, and fully connected layers using libraries like `torch.nn` or `keras.layers`.  

The team will then pursue transfer learning with architectures trained on ImageNet such as **ResNet50**, **EfficientNetB0**, and **VGG16**. These would be implemented through their respective functions:  

- `torchvision.models.resnet50`  
- `tensorflow.keras.applications.EfficientNetB0`  
- `keras.applications.vgg16.VGG16`  

Residual networks such as ResNet make very deep models trainable and effective, providing a strong structure for fine-tuning on medical images [1]. Additionally, surveys of medical image analysis show that transfer learning generally improves performance when there is limited amount of labeled data [2], which is something that commonly happens in MRI datasets.  

To wrap up the methodology, the team will compare a traditional baseline using engineered features like **HOG** or **PCA**, followed by a **Random Forest Classifier**. This procedure will allow the team to understand the differences in performance between deep-learning and classical baseline approaches.  

## Supervised Learning

The problem will be framed as a supervised multi-class classification task with four labels: **glioma, meningioma, pituitary, and healthy**.  

The team will train models on labeled MRI images and optimize **categorical cross-entropy loss** with the **Adam optimizer**.  

Model evaluation will use **accuracy, F-score, and confusion matrices** to ensure reliable clinical performance.  


# Results

We will evaluate the performance of our brain tumor classification models using multiple quantitative metrics. We will strive for a high accuracy in predictions while precision, recall, and F score will ensure that performance is balanced across all tumor categories. In addition, we will use the area under the ROC curve to measure separability between classes and confusion matrices to identify specific error patterns. 

Our target is to achieve at least 80% accuracy with consistent precision and recall across the four classes: glioma, meningioma, pituitary, and no tumor. Meeting these benchmarks will demonstrate the model’s robustness and reduce the risk of bias toward particular tumor types. Our broader goal is to create a lightweight model suitable for hospital environments that functions as a decision support tool for radiologists. Ethical considerations will remain central, ensuring that the model supports rather than replaces clinical judgment. 

We anticipate that transfer learning architectures such as ResNet will outperform baseline CNNs, as demonstrated in prior studies on medical imaging [3]. With proper preprocessing and training, our system should provide reliable second opinions, helping radiologists reduce diagnostic oversights and improving patient outcomes. 


# References

[1] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. CVPR, 2016. 

[2] G. Litjens, T. Kooi, B. E. Bejnordi, et al., “A Survey on Deep Learning in Medical Image Analysis,” Medical Image Analysis, vol. 42, pp. 60–88, 
2017.tworks,” Proc. IEEE ICIP, 2018, pp. 3129–3133. 

[3] A. Afshar, A. Mohammadi, and K. Plataniotis, “Brain tumor type classification via capsule networks,” IEEE ICIP, 2018, pp. 3129–3133.

# Contributions

| Name                  | Proposal Contribution                |
|-----------------------|--------------------------------------|
| **Colin Shaw**        | Github repo/pages and problem definition |
| **Eduardo Romero Serra** | Team Organization / Methods / Presentation and Video|
| **Vinayak Ramasubramanian** | Topic research & ideation / Results / Markdown formatting|
| **Xingjian Ren**      | Gantt Chart|
| **Matthew Sampt**     | Introduction / Markdown Formatting|

[# Gantt Chart](https://docs.google.com/spreadsheets/d/1DeXpFdrviHhOgzM-KoJsPBr04g7CaRDW/edit?usp=sharing&ouid=112407754076113639711&rtpof=true&sd=true)

