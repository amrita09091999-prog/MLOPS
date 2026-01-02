import pandas as pd
import os 
import logging
import yaml
from sqlalchemy import create_engine
from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import numpy as np
import pickle
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logs_dir = '/Users/amritamandal/Desktop/Python/MLOPS/DVC_AWS/MLOPS-1/logs'
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger('model_building_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(logs_dir, 'model_building_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str):
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug(f"Parametes retrieved from params - {params_path}")
        return params 
    except FileNotFoundError:
            logger.debug(f"Params folder doesnt exist - {params_path}")
            raise
    except yaml.YAMLError as e:
            logger.error('YAML error: %s', e)
            raise
    except Exception as e:
            logger.error('Unexpected error: %s', e)
            raise

def load_data(table_name, engine):
    query = f"""select * from {table_name};"""
    try:
        data = pd.read_sql_query(query,engine)
        logger.debug(f"Data has been read from the database  - {table_name}")
    except Exception as e:
        logger.error(f"Unexpected error - {e}")
        raise

    return data

def get_models():
    try:
        params = load_params('params.yaml')
        rf_params = params['random_forest']
        xgb_params = params['xgboost']
        lgb_params = params['lightgbm']

        models = {
        "random_forest": RandomForestRegressor(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_leaf=rf_params['min_samples_leaf'],
            n_jobs=rf_params['n_jobs'],
            random_state=rf_params['random_state']
        ),

        "xgboost": XGBRegressor(
            n_estimators=xgb_params['n_estimators'],
            learning_rate=xgb_params['learning_rate'],
            max_depth = xgb_params['max_depth'],
            subsample=xgb_params['subsample'],
            colsample_bytree=xgb_params['colsample_bytree'],
            objective=xgb_params['objective'],
            tree_method=xgb_params['tree_method'],
            random_state=xgb_params['random_state']
        ),

        "lightgbm": lgb.LGBMRegressor(
            objective=lgb_params['objective'],
            n_estimators=lgb_params['n_estimators'],
            learning_rate=lgb_params['learning_rate'],
            num_leaves=lgb_params['num_leaves'],
            min_data_in_leaf=lgb_params['min_data_in_leaf'],
            subsample=lgb_params['subsample'],
            colsample_bytree=lgb_params['colsample_bytree'],
            random_state=lgb_params['random_state']
        )
        } 
        logging.debug("Model parameters have been assigned")
        return models 
    except Exception as e:
        logger.error(f"Unexpected error - {e}")
        raise

def evaluate(y_true, y_pred,features_num):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2,
        "Adj R2":1 - (1 - r2) * (n - 1) / (n - features_num - 1)
    }

def train_and_evaluate(X_train, X_val, Y_train, Y_val):
    params = load_params('params.yaml')
    selection_metric = params['selection_metric']['metric']

    results = {}
    models = get_models()

    best_model = None
    best_model_name = None
    best_metric_value = -np.inf  # because higher is better (R2 / Adj R2)

    for model_name, model in models.items():
        model.fit(X_train, Y_train)
        logger.debug(f"Model trained - {model_name}")

        Y_pred = model.predict(X_val)
        evaluation = evaluate(Y_val, Y_pred, len(X_train.columns))

        results[model_name] = {
            "model_name": model_name,
            "metrics": evaluation
        }

        logger.debug(
            f"""Model: {model_name}
            MAE: {evaluation['MAE']}
            RMSE: {evaluation['RMSE']}
            R2: {evaluation['R2']}
            Adj R2: {evaluation['Adj R2']}
            """
        )

        # Track best model
        metric_value = evaluation[selection_metric]
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_model = model
            best_model_name = model_name

    logger.debug(
        f"Best model -> {best_model_name} -> metrics {results[best_model_name]['metrics']}"
    )

    return best_model, best_model_name, results


def save_model_metrics(model,results,file_path:str):
    try:
        os.makedirs(file_path, exist_ok = True)
        with open(os.path.join(file_path,"best_model.pkl"),"wb") as f:
            pickle.dump(model,f)
        with open(os.path.join(file_path,'model_eval_metrics.json'),'w') as f:
            json.dump(results,f)
        logger.debug(f"Model has been saved -> {file_path}/best_model.pkl")
        logger.debug(f"Evaluation summary has been saved  ->{os.path.join(file_path),'model_eval_metrics.json'}")
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        engine = create_engine(
        "postgresql+psycopg2://postgres:1234@localhost:5432/postgres"
        )
        train_data = load_data("weather_aqi_processed_train", engine)
        val_data = load_data("weather_aqi_processed_val", engine)

        X_train = train_data.drop(columns = ['pm2_5_mean_t+1'])
        X_val = val_data.drop(columns = ['pm2_5_mean_t+1'])
        Y_train = train_data['pm2_5_mean_t+1']
        Y_val = val_data['pm2_5_mean_t+1']

        features_num = len(X_train.columns)
        best_model, best_model_name, validation_results = train_and_evaluate(X_train, X_val, Y_train, Y_val)

        file_path = "/Users/amritamandal/Desktop/Python/MLOPS/DVC_AWS/MLOPS-1/model_metrics"
        save_model_metrics(best_model, validation_results, file_path)
        logger.debug(f"Final selected model: {best_model_name}")
    
    except Exception as e:
        logger.debug(f"Error has occured - {e}")
        raise

if __name__ == '__main__':
    main()












