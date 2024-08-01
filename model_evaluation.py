from sklearn.metrics import classification_report
import pickle

# Load the model and scaler
with open('fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Preprocess test data
X_test = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
