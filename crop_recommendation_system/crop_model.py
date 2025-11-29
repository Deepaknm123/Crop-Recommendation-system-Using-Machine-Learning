import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Split the data into features and target
X = data.drop('label', axis=1)
y = data['label']

# Label encode target for XGBoost
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# First split the original y into train/test
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

# Then encode the y labels after splitting
y_train = le.transform(y_train_raw)
y_test = le.transform(y_test_raw)





# Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
# model.fit(X_train, y_train)
model.fit(X_train, y_train_raw)

# svm 
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
# svm_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train_raw)


# Train XGBoost
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=10,
    max_depth=2,
    learning_rate=0.2,
    random_state=42
)
# xgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)



# Save the model to a file
with open('crop_recommendation_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save SVM model
with open('svm_model.pkl', 'wb') as svm_file:
    pickle.dump(svm_model, svm_file)

# Save xgboost model
with open('xgb_model.pkl', 'wb') as xgb_file:
    pickle.dump(xgb_model, xgb_file)


print("Model trained and saved as crop_recommendation_model.pkl")

print("✅ SVM model saved as 'svm_model.pkl'")

print("✅ XGBoost model saved as 'xgb_model.pkl'")
