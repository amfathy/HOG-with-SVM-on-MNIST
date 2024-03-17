import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
features = digits.images
labels = digits.target

n_samples = len(features)
data = features.reshape((n_samples, -1))

def compute_gradients(image):
    dx = np.zeros_like(image)
    dy = np.zeros_like(image)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            dx[i, j] = image[i, j + 1] - image[i, j - 1]
            dy[i, j] = image[i + 1, j] - image[i - 1, j]

    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx)

    return magnitude, orientation

def compute_hog(image, cell_size=(4, 4), num_bins=8):
    magnitude, orientation = compute_gradients(image)

    cell_height, cell_width = cell_size
    num_cells_x = image.shape[1] // cell_width
    num_cells_y = image.shape[0] // cell_height

    hog_features = []

    for y in range(num_cells_y):
        for x in range(num_cells_x):
            cell_magnitude = magnitude[y*cell_height:(y+1)*cell_height,
                                       x*cell_width:(x+1)*cell_width]
            cell_orientation = orientation[y*cell_height:(y+1)*cell_height,
                                           x*cell_width:(x+1)*cell_width]

            hist, _ = np.histogram(cell_orientation, bins=num_bins, range=(0, 2*np.pi), weights=cell_magnitude)
            hog_features.extend(hist)

    return hog_features

hog_features = []
for img in features:
    hog_features.append(compute_hog(img))

hog_features = np.array(hog_features)

# Normalize features
scaler = StandardScaler()
hog_features_scaled = scaler.fit_transform(hog_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features_scaled, labels, test_size=0.2, random_state=42)

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predictions
y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)

# Evaluate classifier
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
