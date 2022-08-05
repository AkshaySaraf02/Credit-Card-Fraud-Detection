import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


# Load the Credit Card Data into a variable
card_data = pd.read_csv("C:\\Users\\INDIANS\\Desktop\\Akshay's Projects\\Credit Card Fraud Detection\\creditcard.csv")


# Splitting the Data into Legit and Fraud Transactions
legit = card_data[card_data.Class == 0]
fraud = card_data[card_data.Class == 1]


# Data Preprocessing and Analyzing
print(legit.shape)
print(fraud.shape)

legit.Amount.describe()
fraud.Amount.describe()

card_data.groupby('Class').mean()


# Making a new legit data where the data is low so as to match the fraud data to train the model well
legit_sample = legit.sample(n=492)


# Now concatenate two data to make a new data set
new_data = pd.concat([legit_sample, fraud], axis=0)


# Assigning the data into X and y values
X = new_data.drop(columns="Class", axis=1)  # As X is nothing but the dataset without the Class column.
y = new_data['Class']


# Train Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)


# Defining the model and Fitting the data
model = LogisticRegression()
model.fit(X_train, y_train)


# Now predicting the values
predictions = model.predict(X_test)


# Now finally printing the Predictions, Actual Values and Accuracy
print("Predictions", predictions)
print("Original", y_test)
print(confusion_matrix(y_test, predictions), classification_report(y_test, predictions))
