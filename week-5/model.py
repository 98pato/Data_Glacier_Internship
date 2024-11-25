import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv('iris.csv')

print(df.head())

# Select independent and dependent variables
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
Y = df['species']

# Split the dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

# Feature scaling for independent variables only
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  

# Encode the target variable
le = LabelEncoder()
Y_train = le.fit_transform(Y_train) 
Y_test = le.transform(Y_test)        

# Instantiate the model
classifier = RandomForestClassifier(random_state=50)

# Fit the model
classifier.fit(X_train, Y_train)

# Save the trained model as a pickle file
pickle.dump(classifier, open("model.pkl", "wb"))


