import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the Wine dataset
data = load_wine()
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize classifiers
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)

# Hyperparameter grid for Decision Tree
dt_param_grid = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search for best hyperparameters for Decision Tree
dt_grid_search = GridSearchCV(dt, dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
dt_grid_search.fit(X_train, y_train)

# Perform grid search for best hyperparameters for Random Forest
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Get the best models from grid search
dt_best_model = dt_grid_search.best_estimator_
rf_best_model = rf_grid_search.best_estimator_

# Evaluate models using cross-validation
dt_cv_scores = cross_val_score(dt_best_model, X_train, y_train, cv=5, scoring='accuracy')
rf_cv_scores = cross_val_score(rf_best_model, X_train, y_train, cv=5, scoring='accuracy')

# Print cross-validation results
print(f"Decision Tree Cross-Validation Accuracy: {np.mean(dt_cv_scores) * 100:.2f}%")
print(f"Random Forest Cross-Validation Accuracy: {np.mean(rf_cv_scores) * 100:.2f}%")

# Test the models on the test set
dt_test_predictions = dt_best_model.predict(X_test)
rf_test_predictions = rf_best_model.predict(X_test)

# Evaluate final accuracy on test set
dt_test_accuracy = accuracy_score(y_test, dt_test_predictions)
rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)

# Print test set accuracy
print(f"Decision Tree Test Accuracy: {dt_test_accuracy * 100:.2f}%")
print(f"Random Forest Test Accuracy: {rf_test_accuracy * 100:.2f}%")

# Plotting the accuracy comparison between Decision Tree and Random Forest
models = ['Decision Tree', 'Random Forest']
accuracies = [dt_test_accuracy * 100, rf_test_accuracy * 100]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.title("Accuracy Comparison between Decision Tree and Random Forest")
plt.ylabel("Accuracy (%)")
plt.ylim([0, 110])
plt.show()

# Confusion Matrix for Decision Tree
dt_cm = confusion_matrix(y_test, dt_test_predictions)
dt_disp = ConfusionMatrixDisplay(confusion_matrix=dt_cm, display_labels=data.target_names)
dt_disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Decision Tree")
plt.show()

# Confusion Matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_test_predictions)
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=data.target_names)
rf_disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Random Forest")
plt.show()
