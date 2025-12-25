# Computer Vision: Advanced Image Processing & Deep Learning

This repository showcases a comprehensive implementation of foundational and advanced computer vision algorithms, covering classical image processing techniques and modern deep learning approaches. The projects demonstrate expertise in multi-resolution analysis, geometric transformations, feature detection, image alignment, and generative models—core competencies for research in signal processing, telecommunications, and AI-driven visual systems.

## Table of Contents
- [Overview](#overview)
- [Lab Projects](#lab-projects)
  - [Exercise 1: Multi-Scale Pyramids & Semantic Segmentation](#exercise-1-multi-scale-pyramids--semantic-segmentation)
  - [Exercise 2: Geometric Transformations & Image Warping](#exercise-2-geometric-transformations--image-warping)
  - [Exercise 3: SIFT Feature Detection & RANSAC](#exercise-3-sift-feature-detection--ransac)
  - [Exercise 4: ECC Image Alignment & Visual Tracking](#exercise-4-ecc-image-alignment--visual-tracking)
  - [Exercise 5: Autoencoders & Variational Learning](#exercise-5-autoencoders--variational-learning)
- [Technical Stack](#technical-stack)
- [Key Achievements](#key-achievements)
- [Installation](#installation)
- [Usage](#usage)

## Overview

This work addresses critical challenges in computer vision through rigorous mathematical foundations and efficient algorithmic implementations. Each project explores signal processing concepts essential for modern applications in autonomous systems, medical imaging, telecommunications, and intelligent visual analysis.

**Research-Oriented Focus:**
- **Multi-Resolution Analysis**: Gaussian/Laplacian pyramids for efficient image representation and frequency decomposition
- **Robust Estimation**: RANSAC-based outlier rejection for reliable feature matching under noise
- **Optimization Theory**: Gradient-descent alignment algorithms (ECC, Lucas-Kanade) for sub-pixel registration
- **Generative Modeling**: Variational autoencoders with latent space manifold learning
- **Deep Learning**: Pyramid pooling architectures (PSPNet) for semantic scene understanding

The implementations balance theoretical rigor with computational efficiency, making them suitable for real-time systems and resource-constrained environments—critical requirements in telecommunications hardware and embedded vision systems.

---

## Lab Projects

### Exercise 1: Multi-Scale Pyramids & Semantic Segmentation

**Technologies:** MATLAB, Python (PyTorch), PSPNet  
**Techniques:** Gaussian/Laplacian Pyramids, Spatial Pyramid Pooling, Semantic Segmentation

This exercise implements multi-resolution image analysis through pyramid decompositions and applies pyramid-based architectures to semantic segmentation.

#### Part A: Classical Pyramid Processing (MATLAB)

**Core Implementations:**
- **`genPyr.m`**: Generates Gaussian and Laplacian pyramids with arbitrary levels for multi-scale representation
- **`pyrBlend.m`**: Implements seamless image blending using Laplacian pyramid fusion with Gaussian-weighted masks
- **`pyrReconstruct.m`**: Reconstructs images from Laplacian decomposition
- **`pyr_reduce.m` / `pyr_expand.m`**: Efficient pyramid downsampling/upsampling with anti-aliasing

**Key Applications:**
- Image fusion and compositing with perceptually smooth transitions
- Frequency-domain image editing and analysis
- Multi-scale feature extraction for object detection

#### Part B: Deep Learning Segmentation (PyTorch)

**Implementation:** `pspnet_eval.py`

Evaluates a pre-trained **Pyramid Scene Parsing Network (PSPNet)** on the Cityscapes dataset, demonstrating modern applications of pyramid concepts in deep learning.

**Technical Highlights:**
- Spatial pyramid pooling module aggregating context at multiple scales (1×1, 2×2, 3×3, 6×6)
- Per-class IoU computation for 35 semantic categories
- Proper handling of class imbalance in urban scene parsing
- GPU-accelerated inference with CUDA support

**Research Relevance:**  
Pyramid pooling architectures are critical in applications requiring spatial context understanding (autonomous driving, medical imaging, satellite analysis). The multi-scale aggregation enables robust recognition under varying object sizes—a common challenge in telecommunications signal processing and radar imaging.

**Files:** `er1_final.m`, `er2_final.m`, `er3_final.m`, `pspnet_eval.py`

---

### Exercise 2: Geometric Transformations & Image Warping

**Technologies:** MATLAB  
**Concepts:** Affine Transformations, Homographies, Spatial Interpolation

This exercise explores parametric image transformations essential for camera calibration, image registration, and augmented reality applications.

#### Implemented Transformations:

**1. Linear Scaling (Task 2):**
- Multi-scale image composition with precise spatial alignment
- Applications: Thumbnail generation, multi-resolution displays

**2. Rotation (Task 3):**
- Arbitrary angle rotation with center-point specification
- Bilinear interpolation for sub-pixel accuracy

**3. Shearing (Task 4):**
- Horizontal and vertical shear transformations
- Applications: Perspective correction, slant removal in document imaging

**4. Euclidean Transformations (Task 5):**
- Combined rotation + translation (rigid body motion)
- Critical for motion estimation in video processing

**5. Affine Transformations (Task 6):**
- Full 6-parameter affine model (rotation, translation, scaling, shearing)
- Used in image mosaicking and panorama stitching

**6. Homographies (Task 7):**
- Projective transformations for planar surface mapping
- Essential for camera pose estimation and 3D reconstruction

**7. Image Mosaicking (Task 8):**
- Multi-image fusion using homography estimation
- Applications: Panoramic photography, wide-field-of-view imaging

**Technical Implementation:**
- Custom matrix transformation pipelines
- Boundary handling and interpolation strategies
- Visualization of transformation effects

**Research Applications:**  
These transformations are fundamental in telecommunications (antenna array processing), medical imaging (CT/MRI registration), and satellite imaging (multi-temporal analysis). The implementations demonstrate understanding of homogeneous coordinates and projective geometry—mathematical frameworks shared with radar signal processing and synthetic aperture imaging.

**Files:** `Ex2_Task2.m` through `Ex2_Task8.m`

---

### Exercise 3: SIFT Feature Detection & RANSAC

**Technologies:** MATLAB  
**Algorithms:** SIFT, RANSAC, Robust Feature Matching

This exercise implements scale-invariant feature detection and robust geometric estimation—core techniques for object recognition, image retrieval, and visual odometry.

#### Core Components:

**1. SIFT Feature Extraction (`SIFT_feature.m`):**
- Scale-space extrema detection using Difference-of-Gaussian (DoG) pyramids
- Keypoint localization with sub-pixel refinement
- Orientation assignment for rotation invariance
- 128-dimensional descriptor generation from gradient histograms

**Technical Details:**
- Multi-octave pyramid construction (σ₀ = √2, 3 levels per octave)
- Extrema detection in 3D scale-space (26-neighbor comparison)
- Contrast and edge response filtering for stability

**2. RANSAC Matching (`SIFT_RANSAC_Matching.m`):**
- Descriptor-based feature correspondence using ratio test (threshold: 0.8)
- Outlier rejection via RANSAC with projective transformation model
- Homography estimation from inlier correspondences

**Applications:**
- Image matching under rotation, scaling, and illumination changes
- Object recognition in cluttered scenes
- Camera motion estimation for visual SLAM

**Research Significance:**  
SIFT remains the gold standard for feature-based matching in scenarios requiring robustness to geometric and photometric distortions. RANSAC's iterative outlier rejection mirrors robust estimation techniques in telecommunications (interference rejection, channel estimation) and radar processing (clutter suppression).

**Files:** `SIFT_feature.m`, `SIFT_RANSAC_Matching.m`, `rotation.m`

---

### Exercise 4: ECC Image Alignment & Visual Tracking

**Technologies:** MATLAB, Simulink  
**Algorithms:** Enhanced Correlation Coefficient (ECC), Lucas-Kanade, Image Jacobian

This exercise implements state-of-the-art gradient-based image alignment—essential for motion estimation, video stabilization, and object tracking.

#### Part A: Alignment Algorithm Implementation

**Core Functions:**

**1. `ecc_lk_alignment.m`:**
- Simultaneous ECC and Lucas-Kanade (LK) alignment for comparative analysis
- Multi-level pyramid optimization for coarse-to-fine registration
- Supports affine and homography transformation models

**2. `spatial_interp.m`:**
- Inverse warping with bilinear interpolation
- Handles homogeneous coordinate transformations

**3. `image_jacobian.m`:**
- Computes Jacobian matrix ∂I/∂p of warped image wrt transformation parameters
- Essential for gradient-based optimization

**4. `warp_jacobian.m`:**
- Analytical Jacobian of warp function wrt parameters
- Enables efficient Gauss-Newton optimization

**5. `param_update.m`:**
- Parameter update step in iterative alignment
- Handles compositional and additive update schemes

#### Part B: Simulink Visual System Models

Real-time implementation of vision algorithms using Simulink block diagrams:

**Models:**
- **Corner Detection** (`corner_detection_2019b.mdl`): Harris corner detector
- **Geometric Transformations**: Shear, affine, projective warping modules
- **Mosaic Generation** (`mosaic_2019b.mdl`): Real-time image stitching pipeline

**Research Applications:**  
ECC alignment maximizes correlation in the frequency domain—a concept directly applicable to communications (synchronization, carrier recovery) and radar (pulse compression). The Lucas-Kanade framework is foundational in optical flow estimation, critical for video compression and autonomous navigation.

**Files:** `ecc_lk_alignment.m`, `image_jacobian.m`, `spatial_interp.m`, `warp_jacobian.m`, `param_update.m`, plus Simulink models

---

### Exercise 5: Autoencoders & Variational Learning

**Technologies:** Python (PyTorch), Deep Learning  
**Architectures:** Linear Autoencoders, Convolutional Autoencoders, Variational Autoencoders (VAE)

This exercise explores unsupervised representation learning through autoencoder architectures, comparing learned features to classical dimensionality reduction (PCA) and investigating probabilistic generative models.

#### Implemented Models:

**1. Linear Autoencoder (`autoencoder_mnist_1.py`):**
- Single-layer encoder/decoder (784 → 128 → 784)
- No biases (mimics PCA structure)
- Xavier weight initialization
- **Key Analysis**: Cosine similarity tracking between learned encoder weights and PCA eigenvectors
- **Finding**: Converges to PCA subspace, validating theoretical equivalence

**2. Multi-Layer Autoencoder (`autoencoder_mnist_2.py`):**
- Deep architecture: 784 → 256 → 128 → 256 → 784
- ReLU activations for non-linear feature learning
- Surpasses PCA in reconstruction quality (lower MSE)

**3. Convolutional Autoencoder (`autoencoder_mnist_3.py`):**
- Spatial feature extraction via conv layers
- Preserves local image structure
- Efficient for translation-invariant representations

**4. Variational Autoencoder - VAE (`VAE.py`):**
- Probabilistic latent space (μ, σ) with reparameterization trick
- KL divergence regularization for continuous latent manifold
- 2D latent space visualization showing digit clusters
- Generative sampling from learned distribution

**Training Details:**
- Optimizer: Adam (lr=1e-3)
- Loss: Binary cross-entropy (reconstruction) + KL divergence (VAE)
- Batch size: 250, Epochs: 40-100
- Dataset: MNIST (60K train, 10K test)

#### Comparative Analysis:
- **Experiment 1**: Linear AE vs PCA (weight similarity convergence)
- **Experiment 2**: Reconstruction error comparison across architectures
- **Experiment 3**: Latent space interpolation and manifold structure
- **Experiment 4**: VAE generative capabilities with fixed noise

**Research Relevance:**  
Autoencoders are core to modern signal processing (denoising, compression, anomaly detection). VAE's probabilistic framework mirrors Bayesian estimation in communications and radar. The learned latent representations enable efficient signal encoding—critical for bandwidth-limited telecommunications channels.

**Files:**  
- `autoencoder_mnist_1.py` through `autoencoder_mnist_4.py`
- `VAE.py`, `compare_autoencoders.py`
- Jupyter notebooks: `autoencoder_mnist_*.ipynb`, `VAE.ipynb`

---

## Technical Stack

### Core Technologies
- **MATLAB R2019b+**
  - Image Processing Toolbox
  - Computer Vision Toolbox
  - Simulink (for real-time modeling)

- **Python 3.8+**
  - PyTorch 1.10+ (deep learning framework)
  - NumPy, SciPy (numerical computing)
  - OpenCV (optional, for additional processing)
  - scikit-learn (PCA, preprocessing)
  - Matplotlib (visualization)

### Algorithms Implemented
- **Classical CV**: SIFT, RANSAC, Harris corners, pyramid decomposition
- **Alignment**: ECC, Lucas-Kanade, Gauss-Newton optimization
- **Deep Learning**: Autoencoders (linear, CNN, variational), PSPNet
- **Transformations**: Affine, projective, Euclidean mappings
- **Optimization**: Gradient descent, compositional updates

### Mathematical Foundations
- Multi-scale signal representation
- Robust statistical estimation
- Differential geometry (Jacobians, Hessians)
- Variational inference
- Information theory (KL divergence, mutual information)

---

## Installation

### Prerequisites

**MATLAB:**
```bash
# Required Toolboxes:
# - Image Processing Toolbox
# - Computer Vision Toolbox
# - Simulink (for Exercise 4, Part B)
```

**Python Environment:**
```bash
# Create virtual environment
python -m venv cv_env
source cv_env/bin/activate  # Windows: cv_env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib
pip install scikit-learn
pip install opencv-python  # optional
```

### Dataset Setup

**Exercise 1 (PSPNet):**
- Download Cityscapes dataset from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)
- Download PSPNet checkpoint: `train_epoch_200_CPU.pth`

**Exercise 5 (Autoencoders):**
```bash
# Download MNIST CSV files
# From: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
# Files needed:
#   - mnist_train.csv
#   - mnist_test.csv
```

---

## Usage

### Exercise 1: Pyramids

**MATLAB:**
```matlab
% Laplacian pyramid blending
run('ex01/Ex01-Katsaros_Andreas/Matlab_Ex1/pyrBlend.m')

% Individual pyramid generation
img = imread('apple1.jpg');
pyr = genPyr(img, 'gauss', 5);  % 5-level Gaussian pyramid
```

**Python (PSPNet):**
```bash
cd "ex01/Ex01-Katsaros_Andreas/PspNet"
python pspnet_eval.py
# Evaluates model on Cityscapes validation set
# Outputs per-class IoU metrics
```

### Exercise 2: Transformations

```matlab
% Run individual tasks
cd "ex02/Ex02-Katsaros_Andreas/CV_2-TRANSFORMATIONS"
run('Ex2_Task2.m')  % Scaling
run('Ex2_Task3.m')  % Rotation
run('Ex2_Task7.m')  % Homography
run('Ex2_Task8.m')  % Mosaicking
```

### Exercise 3: SIFT + RANSAC

```matlab
cd "Ex03-Katsaros_Andreas/CV_3-SIFT"
% Feature extraction
run('SIFT_feature.m')

% Matching with RANSAC
run('SIFT_RANSAC_Matching.m')
```

### Exercise 4: Image Alignment

```matlab
cd "ex04/Ex04-Katsaros_Andreas/Part A"

% Load images
template = imread('template.jpg');
image = imread('image.jpg');

% Perform alignment
[results, results_lk, MSE, rho, MSELK] = ecc_lk_alignment(image, template, 3, 50, 'affine', []);

% results.warp contains final transformation
% rho = ECC correlation coefficient
```

### Exercise 5: Autoencoders

```bash
cd "EX05/work"

# Linear autoencoder (comparison with PCA)
python autoencoder_mnist_1.py

# Deep autoencoder
python autoencoder_mnist_2.py

# Convolutional autoencoder
python autoencoder_mnist_3.py

# Variational autoencoder
python VAE.py

# Or run Jupyter notebooks:
jupyter notebook autoencoder_mnist_1.ipynb
```

---

## Institution

**University of Patras**  
**Department of Electrical and Computer Engineering**  
**Student ID:** 1084522  
**Course:** Computer Vision Laboratory

---

## Contact

For research collaboration opportunities in computer vision, signal processing, or AI systems, feel free to connect.

**Email:** andreaskatarosgr@gmail.com

---

*This repository demonstrates applied research capabilities in computer vision, suitable for positions in signal processing R&D, telecommunications research, AI labs, and vision systems development.*
