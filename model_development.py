from sklearn.ensemble import RandomForestClassifier
import pickle

# Define and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(model, file)
