import pandas as pd
from pycaret.classification import predict_model, load_model
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """
    Loads churn data into a DataFrame from a string filepath.
    """
    data = pd.read_csv(filepath)

    # Preprocess 'CustomerID' column if needed
    le = LabelEncoder()
    data['CustomerID_Encoded'] = le.fit_transform(data['CustomerID'])
    data['CustomerID'] = data['CustomerID_Encoded']
    data.drop(columns=['CustomerID_Encoded'], inplace=True)

    return data

def make_predictions(data, threshold=0.5):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    Rounds up to 1 if greater than or equal to the threshold.
    """
    predictions = predict_model(model, data=data)
    predictions['Churn_prediction'] = (predictions['Churn_Numeric'] >= threshold)
    predictions['Churn_prediction'].replace({True: 'Churn', False: 'No churn'}, inplace=True)
    drop_cols = predictions.columns.tolist()
    drop_cols.remove('Churn_prediction')
    return predictions.drop(drop_cols, axis=1)

if __name__ == "__main__":
    model = load_model('LinDiscrAnalysis')  # Assuming the model is named 'LinDiscrAnalysis'
    data = load_data('new_churn_data.csv')  # Replace with your data path
    predictions = make_predictions(data)
    print('predictions:')
    print(predictions)