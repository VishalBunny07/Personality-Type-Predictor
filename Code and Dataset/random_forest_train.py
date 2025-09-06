import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('personality_synthetic_dataset.csv')

# Encode the target variable
df['personality_type'] = df['personality_type'].map({'Introvert': 0, 'Extrovert': 1, 'Ambivert': 2})

# Prepare features and target
X = df.drop('personality_type', axis=1)
y = df['personality_type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (using Random Forest as it performed well)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('personality_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the column names for reference
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)