import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
data_file = "data.txt"
data = np.loadtxt(data_file)  # Adjust delimiter if necessary

# Preprocess the data
# Assuming that the last column in the data is the label
X = data[:, :-1]  # All rows, all columns except the last
y = data[:, -1]   # All rows, only the last column

# Split the data into training and testing sets. Stratify ensures same proportion of objects for test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

# Train the model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusion_matrix(y_test, y_pred))

with open("model", "wb") as f:
    pickle.dump(clf, f)