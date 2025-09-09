**Problem:**

A Brain tumor is considered as one of the aggressive diseases, among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous System(CNS) tumors. Every year, around 11,700 people are diagnosed with a brain tumor. The 5-year survival rate for people with a cancerous brain or CNS tumor is approximately 34 percent for men and36 percent for women. Brain Tumors are classified as: Benign Tumor, Malignant Tumor, Pituitary Tumor, etc. Proper treatment, planning, and accurate diagnostics should be implemented to improve the life expectancy of the patients. The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). A huge amount of image data is generated through the scans. These images are examined by the radiologist. A manual examination can be error-prone due to the level of complexities involved in brain tumors and their properties.

**Introduction to Dataset Brain Tumor Classification(MRI) Data**
 Brain_Tumor_Dataset/
 ├── Training/
 │    ├── glioma_tumor/
 │    │     ├── img1.jpg
 │    │     ├── img2.jpg
 │    │     └── ...
 │    ├── meningioma_tumor/
 │    ├── pituitary_tumor/
 │    └── no_tumor/
 │
 └── Testing/
      ├── glioma_tumor/
      ├── meningioma_tumor/
      ├── pituitary_tumor/
      └── no_tumor/
So the classes are :glioma_tumor, meningioma_tumor, pituitary_tumor, 

**About Dataset: Brain Tumor MRI Dataset**
This dataset is a combination of the following three datasets :
figshare
SARTAJ dataset
Br35H
This dataset contains 7023 images of human brain MRI images which are classified into 4 classes: glioma - meningioma - no tumor and pituitary.no tumor class images were taken from the Br35H dataset.

**MODEL 1-Baseline CNN: **

This is our base line CNN model constructed from the scratch. Here is the model structure.This is a CNN with 3 convolutional blocks (Conv → ReLU → MaxPool).
	•	Each pooling step reduces spatial size: 224 → 112 → 56 → 28.
	•	Classifier head: Flatten → Dense(256) → Dropout → Dense(num_classes).
	•	Outputs probabilities for each tumor class.

Loss & Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
	•	Loss: Cross-entropy (good for multi-class classification).
	•	Optimizer: Adam with learning rate 10^(-3)

MODEL TRAINING:
 
This is an image classification project.Here are the steps: 

	•	We use nn.CrossEntropyLoss() for multi-class classification with integer class Labels.
	•	out.argmax(1)-> picks the predicted class.   
	•	It reports accuracy, a classification metric.
	•	The model’s final layer must output logits with shape [batch, num_classes].

**Pre-Trained Model RESNET50:**
This script trains and evaluates a ResNet-50 model on a Brain Tumor MRI dataset using PyTorch. It applies transfer learning in feature extraction mode:
	1	Device Setup – Uses GPU if available, otherwise CPU.
	2	Data Preparation – Applies preprocessing and augmentation (resize, flip, rotation, normalization) for training data, and normalization for test data.
	3	Model Setup – Loads a pretrained ResNet-50, freezes all its convolutional layers, and replaces the final fully connected layer with a new classifier for the brain tumor classes.
	4	Training – Trains only the new classifier layer using CrossEntropyLoss and the Adam optimizer.
	5	Validation & Testing – Evaluates performance on the test set during training, then computes final test accuracy after training.

**Fine tunning ResNet50:**

	1	Two-phase training (warm-up + fine-tune):
	◦	Starts by training only the new classifier head for stability.
	◦	Then unfreezes the last ResNet block (layer4) for deeper fine-tuning with a lower learning rate.
	◦	This approach prevents catastrophic forgetting and improves generalization.
	2	Balanced optimization:
	◦	Uses AdamW with different learning rates for head vs. backbone, which is good practice in transfer learning.
	◦	Includes weight decay to reduce overfitting.
	3	Data preprocessing adapted for MRIs:
	◦	Converts grayscale MRIs to 3 channels (Grayscale(num_output_channels=3)) to match ImageNet weights.
	◦	Applies appropriate normalization with ImageNet mean/std.
	4	Robust training mechanics:
	◦	Integrated ReduceLROnPlateau scheduler to lower LR when validation loss stops improving.
	◦	Includes early stopping and checkpoint saving for the best validation model.
	◦	Supports mixed precision (AMP) for faster GPU training.
	5	Class imbalance handling:
	◦	Optional class weights are computed dynamically from the dataset, reducing bias toward majority classes.


Acknowledgment:  In this Project, I used Chatgpt and Kaggle to do the project. These tools were mostly used to write and debug the code and learning how to design the model.