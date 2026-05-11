import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv("car_data.csv")

# Feature engineering
data['Car_Age'] = 2024 - data['Year']

# Drop unnecessary columns
data.drop(['Car_Name','Year'], axis=1, inplace=True)

# Convert categorical data
data = pd.get_dummies(data, drop_first=True)

# Features and target
X = data.drop(['Selling_Price'], axis=1)
y = data['Selling_Price']

# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train,y_train)

# Save model
pickle.dump(model, open("car_price_model.pkl","wb"))

print("Model saved successfully!")