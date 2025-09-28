import os
import pandas as pd
import numpy as np
import pickle 
import logging
from sklearn.ensemble import RandomForestClassifier

# ensure the 'logs' directory exists
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging configuration 
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler =logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path:str) -> pd.DataFrame:
    """" load data from a csv """

    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed  to  parse  the csv file: %s' , e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray,Y_train: np.ndarray,params: dict) -> RandomForestClassifier:
    """ Train the RandomForestClassifier model.
    :params X_train: Training feature
    :params Y_train: Dictionary of hyperparameters
    :return : Trained RandomForestClassifier """

    try:
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("The number of samples in X_train and Y_train must be the same")
        
        logger.debug('Initializing RandomForest model with patameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.debug('Model training started with %d sample', X_train.shape[0])
        clf.fit(X_train,Y_train)
        logger.debug('Model training completed')

        return  clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('error during model training: %s', e)
        raise

def save_model(model,file_path:str) -> None:
    """save the trained model to a file .
     :param model:Trained model  object
     :param file_path: Path to save the model file """
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path,'wb') as file:
             pickle.dump(model,file)
        logger.debug('Model save to %s', file_path)
    except FileNotFoundError as e:
        logger.error('file path not found: %s',e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = {'n_estimators': 25, 'random_state': 2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train =train_data.iloc[:,:-1].values
        Y_train = train_data.iloc[:,-1].values

        clf = train_model(X_train,Y_train,params)

        model_save_path = 'models/model.pkl'
        save_model(clf,model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error:{e}")

if __name__ == '__main__':
    main()
