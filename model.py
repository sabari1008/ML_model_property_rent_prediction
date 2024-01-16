import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

df = pd.read_csv("Trained_data.csv")


newTrain = df.iloc[:2632]
newTest = df.iloc[1315:]


X = df.drop("rent", axis=1)  # Remove 'rent' column
Y = np.log1p(df["rent"])  # Get 'rent' column (log1p(x) = log(x+1))
print(X.shape)
print(Y.shape)

reg = LinearRegression().fit(X, Y)



# Assuming 'newTest' is your test set
# Get the feature names used during training
training_feature_names = X.columns

# Reorder the columns in newTest to match the order of features in training
newTest = newTest[training_feature_names]

# Make predictions
pred = np.expm1(reg.predict(newTest))

# Create a submission DataFrame
sub = pd.DataFrame({
    'id': newTest.index,  # Assuming the index of 'newTest' serves as an identifier
    'rent': pred
})



# Save the submission to a CSV file
sub.to_csv("submission.csv", index=False)

# Display the submission DataFrame
print(sub)

model_filename = 'model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(reg, file)