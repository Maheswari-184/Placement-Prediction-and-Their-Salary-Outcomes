import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the dataset
data = pd.read_csv("Placement_Data_Full_Class.csv")
data['status'] = data['status'].map({'Not Placed': 0, 'Placed': 1})

# Define features and target variable
X = data[['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p']]
y = data['status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
categorical_features = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), ['ssc_p', 'hsc_p', 'degree_p', 'etest_p'])
    ],
    remainder='passthrough'
)

# Create a pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=300))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)


# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a confusion matrix plot
def create_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Placed', 'Placed'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('static/confusion_matrix.png')
    plt.close()

# Call the function to create and save the confusion matrix plot
create_confusion_matrix(cm)

# Flask application
placeapp = Flask(__name__, template_folder='template')

# Load pre-trained models
model = pickle.load(open('op.pkl', 'rb'))
salary = pickle.load(open('salary.pkl', 'rb'))

# Route for the home page
@placeapp.route('/')
def home():
    return render_template('home.html')

# Route for predictions
@placeapp.route('/predict', methods=['POST'])
def predict():
    features = []
    for i in request.form.values():
        if i in ['Male', 'Female']:
            features.append(0 if i == 'Male' else 1)
        elif i in ['Science', 'Commerce', 'Arts']:
            features.append({'Science': 0, 'Commerce': 1, 'Arts': 2}[i])
        elif i in ['Science and Technology', 'Commerce and Management', 'Architecture and Others']:
            features.append({'Science and Technology': 0, 'Commerce and Management': 1, 'Architecture and Others': 2}[i])
        elif i in ['Yes', 'No']:
            features.append(1 if i == 'Yes' else 0)
        else:
            try:
                features.append(float(i) / 100)  # Scale percentage values
            except ValueError:
                continue

    final_features = np.array(features).reshape(1, -1)  # Reshape for a single prediction
    predictions = model.predict(final_features)
    sal = salary.predict(final_features)

    output = sal[0] * (2 if predictions[0] == 1 else 0.50)
    output = int(output)
    prediction_text = (f'With this track record, YOU WILL BE PLACED! And if you work hard, you will have a chance of getting a package of {output}.00 ₹ per annum'
                       if predictions[0] == 1
                       else f'With this track record, you will NOT be PLACED! And if you work hard and try, you will have a chance of getting a package of {output}.00 ₹ per annum')
    
    return render_template('home.html', prediction_text=prediction_text)

# Route for showing metrics including confusion matrix
@placeapp.route('/metrics')
def show_metrics():
    return render_template('metrics.html', metrics={
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix_image': 'static/confusion_matrix.png'
    })

if __name__ == "__main__":
    placeapp.run(debug=True)
