from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split



dataset = load_breast_cancer()
features = dataset.data
labels = dataset.target

train_features, test_features, train_labels, test_labels = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)
model.fit(train_features, train_labels)

predictions = model.predict(test_features)

accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion matrix:")
print(confusion_matrix(test_labels, predictions))
print("Classification report:")
print(classification_report(test_labels, predictions, target_names=dataset.target_names))
