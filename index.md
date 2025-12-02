# Brain Tumor Classification Project

# Introduction

Brain tumors are abnormal growths within the brain. Tumors cause pressure build-up within the skull, often leading to severe complications and death. Cancerous or not, tumors interfere with critical neurological functions, necessitating timely detection.  

Earlier literature demonstrates that traditional image processing often struggles with categorizing brain tumors due to their wide variability. However, recent developments in deep learning have shown promise in improving accuracy in tumor classification.  

To amass data about tumors, we are using the publicly available [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). This dataset contains over seven thousand Magnetic Resonance Imaging (MRI) scans of human brains. Each image is labeled by its respective tumor category or lack thereof.  

# Problem Definition

Diagnosing brain tumors from medical images is a complex process that is not always foolproof. Our project will give a second opinion for radiologists to ensure that they have made little to no oversights in their diagnosis of the patient into four distinct categories: Glioma, meningioma (non-cancerous), pituitary, and no tumor (healthy). Additionally, classifying the tumor by grade will aid in streamlining patients’ treatment. 

Diagnosing tumors is further complicated by variability in MRI acquisition and tumor appearance, which can lead to inconsistent readings across clinicians. Our system aims to provide a consistent, reproducible second opinion across the four classes, as well as offer a preliminary tumor-grade suggestion to help triage cases and streamline follow-up imaging and treatment planning. 

# Methodology

To classify brain tumors from MRI scans, our team’s approach combines robust preprocessing, supervised deep learning models, and transfer learning to maximize accuracy and generalizability.  

### Data Processing

Data was prepared in two simple steps so that images were consistent and informative for model training:

**Automated cleaning script (OpenCV):** This utility script converts images to grayscale, slightly removes noise with Gaussian blur, removes small artifacts, defines brain region by contours, crops to that region, enhances contrast (histogram equalization + normalization), and resizes to `256×256`. Cleaned copies of the images are then saved to organized class folders.

**Training-time transforms (PyTorch):** When loading data with `torchvision.datasets.ImageFolder`, we apply
- `Grayscale(num_output_channels=1)`
- `Resize((256, 256))`
- `ToTensor()`
- `Normalize(mean=0.5, std=0.5)`

Images are then read from class labeled directories. The team used PyTorch DataLoader with a `BATCH_SIZE = 32` and shuffling for the training set to expose the model to varied batches each epoch.

**Example training images (first: glioma, second: no tumor):**

![Figure 1](src/dataset/Training/glioma/Tr-gl_1232.jpg)\
*Figure 1. Example MRI slice labeled "glioma."*

![Figure 2](src/dataset/Training/notumor/Tr-no_0393.jpg)\
*Figure 2. Example MRI slice labeled "no tumor."*

### Convolutional Machine Learning Models

Three supervised convolutional architectures were evaluated on the same preprocessed dataset and training/testing conditions:

1. **Model 1 – Baseline CNN trained from scratch**  
2. **Model 2 – ResNet18 (transfer learning)**  
3. **Model 3 – EfficientNet-B0 (transfer learning)**

All three models were optimized with cross-entropy loss and the Adam optimizer, and all use the same batch size (32) and image resolution (256×256). This guided more focus towards architectural differences rather than processing or optimization confounds.

#### Model 1 – Baseline Lightweight CNN

A lightweight Convolutional Neural Network (CNN) was implemented to learn special patterns within the MRI image dataset. The model is intentionally small so it trains quickly and runs on modest hardware while still capturing texture and shape cues for a variety of MRI image styles.

**Architecture (PyTorch):**
- Conv blocks (×3): `Conv2d → ReLU → MaxPool(2×2)` with channels `1→16→32→64`, kernel 3×3 (padding 1).
- Flatten to a vector of size `64×32×32`.
- Fully connected: `Linear(64×32×32 → 128) → ReLU → Linear(128 → 4)`.
- The final layer outputs 4 logits (one per class).

**Training setup:**
- Loss: Cross-Entropy, which is standard for multi-class classification.
- Optimizer: Adam with a learning rates of 0.001, 0.005, and 0.01.
- Epochs: 10 for LR = 0.001, 20 for LR = 0.005 & 0.01
- Batch size: 32

To monitor learning, training and evaluation loss and accuracy were printed at each epoch. After training, learned weights were saved as `brain_tumor_cnn.pth` so the classifier could be reloaded for evaluation or for single image predictions without retraining.

This CNN was selected as a baseine because it is easy to understand, fast to train, and well suited for recognizing localized features such as edges, textures, and shapes which are common when distinguishing between tumor types.

#### Model 2 – ResNet18 (transfer learning)

To test a deeper architecture with residual connections, a **ResNet18** model pre-trained on ImageNet was implemented. Residual blocks help gradients flow through deeper networks and can capture more complex features than our small CNN.   

**Key modifications and configuration:**
- **Input adaptation:** The original 3-channel first convolution was modified to accept a **1-channel** grayscale input.  
- **Output head:** The final fully connected layer was replaced with a new linear layer producing **4 class logits**.   
- **Initialization:** Initialized from ImageNet weights and fine-tuned the full network using the brain MRI dataset.
- **Training:**  
  - Loss: Cross-Entropy  
  - Optimizer: Adam with learning rate `0.001`  
  - Epochs: 20  
  - Same `batch_size = 32` and transforms as the baseline CNN   

ResNet18 was selected because it is a widely used residual network that is deeper than the baseline CNN but still lightweight, making it a good candidate for transfer learning on a medium-sized medical imaging dataset.

#### Model 3 – EfficientNet-B0 (transfer learning)

Finally, **EfficientNet-B0** with transfer learning wax implemented. EfficientNet architectures use compound scaling of depth, width, and resolution to provide strong performance with relatively few parameters. This makes EfficientNet-B0 a promising candidate for high accuracy under computational constraints. :contentReference[oaicite:16]{index=16}  

Model customization:

- **Base model:** `torchvision.models.efficientnet_b0(pretrained=True)`  
- **Input adaptation:** Modify the first convolution in the feature extractor to accept a single grayscale channel. :contentReference[oaicite:17]{index=17}  
- **Output head:** Replace the final classifier with a linear layer outputting 4 logits (one per class). :contentReference[oaicite:18]{index=18}  

Training configuration:

- Loss: Cross-Entropy  
- Optimizer: Adam with learning rate `0.001`  
- Epochs: 20  
- Same batch size and transforms as the other models  
- Model weights saved as `brain_tumor_efficientnet_b0.pth` for later evaluation   

EfficientNet-B0 was chosen to represent a **modern, parameter-efficient architecture** that often outperforms older backbones in image classification, especially when fine-tuned on domain-specific data like MRI.

### Classical Model Approach

#### Model 4 – Support Vector Machine (SVM) Classifier

To evaluate a traditional machine learning approach alongside the neural architectures, a **Support Vector Machine (SVM)** classifier was implemented using scikit-learn. Instead of raw MRI pixels, which are high dimensional and difficult for classical models to generalize on, we extracted compressed deep features from the **trained EfficientNet-B0 model** and used them as fixed input representations. This allows the SVM to take advantage of strong CNN-learned feature embeddings while reducing computational complexity and overfitting risk.

**Feature Extraction Workflow**
1. Loaded EfficientNet-B0 model with learned weights fixed.
2. Removed the classification head to output the penultimate feature vector.
3. Passed each MRI image once through EfficientNet-B0 to extract a **high-level feature embedding**.
4. Flattened embeddings and stored them with corresponding class labels for scikit-learn.

This approach transforms all training and testing images into a consistent numerical feature space suitable for classical ML algorithms.

**SVM Classifier Configuration**
- Library: scikit-learn SVC
- Kernel: **RBF**
- Regularization: C = 1.0
- Gamma: “scale”
- Output: One-vs-One decision strategy for multi-class tumor classification

**Why SVM?**
- Provides a **strong margin-based classifier** with proven success on medical imaging feature vectors
- Avoids full end-to-end retraining → **faster experimentation**
- Offers a **reference benchmark** against deep neural models
- Ability to perform well on **smaller or noise-sensitive** feature spaces

This SVM setup tests whether neural network feature extraction combined with a simpler classifier can rival or outperform full deep learning pipelines for brain MRI classification.

# Results

Four approaches were evaluated to conduct brain tumor classification using MRI:  
1) Baseline CNN
2) ResNet18
3) EfficientNet-B0
4) SVM with deep features.

A a dataset of **1,311 images** was used to report all metrics.

### Individual Model Analysis - Baseline CNN

#### Overall Performance

The Accurracy and Preciision/Recall/F1/Support values gathered from running the trained model over a testing set of 1,311 images can bee seen in Table 1. Of the 1,311 testing samples, only 48 showed errors when classifying, with a final **Accuracy:** 96.34%.

*Table 1: Overall performance (Testing set, 1,311 images)*
| Class            | Precision | Recall | F1    | Support |
|------------------|----------:|------:|------:|--------:|
| **glioma**       | 0.952     | 0.927 | 0.939 | 300     |
| **meningioma**   | 0.928     | 0.922 | 0.925 | 306     |
| **no tumor**     | 0.978     | 1.000 | 0.989 | 405     |
| **pituitary**    | 0.990     | 0.993 | 0.992 | 300     |
| **Macro averages** | **0.962** | **0.960** | **0.961** | – |

#### Confidence Summary

For both splits the average prediction confidence is high, with Training 0.9982 ± 0.0142 and Testing 0.9835 ± 0.0638. On the test set, correct predictions are more confident (0.9883 ± 0.0524) than incorrect ones (0.8552 ± 0.1486), which is consistent with a well-behaved classifier. These values can be seen in further detail on Figure 3 below.

![Figure 3](assets/images/Comprehensive_Model_Evaluation_Report.png)
*Figure 3: Summary of accuracy, class-wise metrics, and confidence statistics.*

#### Visualization Results

The confusion matrices in Figure 4 highlight class specific behavior, with **No tumor** and **pituitary** classes being classified almost perfectly. Most errors occur between **glioma** and **meningioma** classes, which are visually similar on some slices, which causes the model to occasionally confuse these two. This visualization reveals that altough the model is mostly accurrate, it has some trouble identifying differences in images that share multiple similarities.

![Figure 4](assets/images/Confusion_Matrices.png)
*Figure 4: Confusion matrices for Training (left) and Testing (right).*

#### Model Perfomance

Overall this model perfomed well and was consistent at identifying and differentiating brain tumor in the majority of cases. Why?
- **Consistent inputs:** Data processing was key to success, with grayscale conversion, normalization, and resizing to 256×256 providing uniform inputs for learning.
- **CNN inductive bias:** Convolutions captured local edges, textures, and shapes that distinguishd tumor types, which fit this task well.
- **Stable optimization:** Minimizing cross-entropy loss with the Adam optimizer provided smooth and efficient convergence.

### Next steps

- **Model extensions:** Add two supervised transfer-learning approaches, fine-tuned ResNet-50 and EfficientNet-B0, for comparison on the same splits. This will help aliviate some of the accurracy issues between **glioma** and **meningioma**.
- **Focused error analysis:** Further prioritize improvements on **glioma** ↔ **meningioma** separability such as targeted preprocessing or feature emphasis, and incorporate probability based evaluation plots to validate improvements.
  
# References

[1] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. CVPR, 2016. 

[2] G. Litjens, T. Kooi, B. E. Bejnordi, et al., “A Survey on Deep Learning in Medical Image Analysis,” Medical Image Analysis, vol. 42, pp. 60–88, 
2017.tworks,” Proc. IEEE ICIP, 2018, pp. 3129–3133. 

[3] A. Afshar, A. Mohammadi, and K. Plataniotis, “Brain tumor type classification via capsule networks,” IEEE ICIP, 2018, pp. 3129–3133.

# Contributions

| Name                  | Proposal Contribution                |
|-----------------------|--------------------------------------|
| **Colin Shaw**        | Data Processing and Preparation |
| **Eduardo Romero Serra** | Team Organization / Report Writting / Gantt Chart |
| **Vinayak Ramasubramanian** | Accurracy / Precision / Visualization |
| **Xingjian Ren**      | Accurracy / Precision / Visualization |
| **Matthew Sampt**     | PyTorch Model Training / Model Testing |

# [Gantt Chart](https://docs.google.com/spreadsheets/d/1DeXpFdrviHhOgzM-KoJsPBr04g7CaRDW/edit?usp=sharing&ouid=112407754076113639711&rtpof=true&sd=true)

