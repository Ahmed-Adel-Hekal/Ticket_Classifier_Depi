import os 
import pandas as pd 
from dotenv import load_dotenv
from metrics import accuracy

load_dotenv()

def load_tickets(path=None) : 
    if path is None : 
        path = os.getenv('TICKETS_DATA_PATH')

    if not path : 
        raise ValueError('Ticket_Data_PATH is not configured')

    return pd.read_csv(path)

def evaluate_predictions(predictions, expected):
    return {
        "accuracy": accuracy(
            expected,
            predictions,
        )
    }

def evaluate_dataset (path=None, predictions=None) : 
    if predictions is None : 
        raise ValueError('Prediction is None')

    df = load_tickets(path)

    expected = df['category'].tolist()

    if len(predictions) != len(expected) : 
        raise ValueError(
            "Numeber of prediction Must Match. \n number of tickets"
        )

    return evaluate_predictions(predictions, expected)