# HOG-with-SVM-on-MNIST
This code trains an SVM classifier using Histogram of Oriented Gradients (HOG) features for handwritten digit classification.


## Steps to Run Handwritten Digit Classification with SVM and HOG

### 1. Load the Dataset
- Load the digits dataset from scikit-learn using `datasets.load_digits()`.

### 2. Define Gradient and HOG Calculation Functions
- Define functions to compute gradients and Histogram of Oriented Gradients (HOG) features for an image.

### 3. Compute HOG Features
- Compute HOG features for each image in the dataset using the previously defined functions.

### 4. Normalize HOG Features
- Normalize the computed HOG features using `StandardScaler` from scikit-learn.

### 5. Split Dataset into Training and Testing Sets
- Split the dataset into training and testing sets using `train_test_split` from scikit-learn.

### 6. Train SVM Classifier
- Train an SVM classifier with a linear kernel on the training data using `SVC(kernel='linear')` from scikit-learn.

### 7. Make Predictions
- Use the trained classifier to make predictions on both training and testing data.

### 8. Evaluate Classifier's Accuracy
- Evaluate the accuracy of the classifier on both the training and testing data using `accuracy_score` from scikit-learn.

