import pandas as pd
import os 
import logging
import yaml
from sqlalchemy import create_engine
from sqlalchemy import text

logs_dir = '/Users/amritamandal/Desktop/Python/MLOPS/DVC_AWS/MLOPS-1/logs'
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(logs_dir, 'data_ingestion.log')
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
    
def load_data(table_name:str,engine):
    query = f"""select * 
    from {table_name};"""
    try:
        data = pd.read_sql_query(query, con=engine)
        logger.debug(f"Table fetched from postgres database, table name - {table_name}")
        return data 
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def assign_splits(row):
    if row['date']<=pd.Timestamp('2025-08-31'):
        return 'train'
    elif row['date'] <= pd.Timestamp('2025-10-31'):
        return 'val'
    else:
        return 'test'

def preprocess_data(data :pd.DataFrame):
    data_sorted= data.sort_values(['state','place','latitude','longitude',"date"]).reset_index(drop=True)
    data_sorted["pm2_5_mean_t+1"] = (
    data_sorted.groupby(['state','place','latitude','longitude'])["pm2_5_mean"]
      .shift(-1)
      )
    data_sorted= data_sorted.dropna(subset=["pm2_5_mean_t+1"])
    data_sorted['date'] = pd.to_datetime(data_sorted['date'])
    data_sorted['split'] = data_sorted.apply(assign_splits, axis=1)
    train_data = data_sorted[data_sorted['split']=='train']
    val_data = data_sorted[data_sorted['split']=='val']
    test_data = data_sorted[data_sorted['split']=='test']
    train_data.drop(columns = ['split'],inplace=True)
    val_data.drop(columns = ['split'],inplace=True)
    test_data.drop(columns = ['split'],inplace=True)

    logger.debug("Preprocessing is done, spliting into train-val-test is done")
    return train_data, val_data, test_data

def save_data(train_data, val_data, test_data, engine):
    try:
        train_data.to_sql(
        name="weather_aqi_train",
        con=engine,
        schema="public",        
        if_exists="replace",     
        index=False
        )
        val_data.to_sql(
        name="weather_aqi_val",
        con=engine,
        schema="public",        
        if_exists="replace",     
        index=False
        )
        test_data.to_sql(
        name="weather_aqi_test",
        con=engine,
        schema="public",        
        if_exists="replace",     
        index=False
        )
        logger.debug("Data saved into database with schema as public")
    except Exception as e:
         logger.error('Unexpected error occurred while saving the data: %s', e)
         raise

def main():
    try:
        engine = create_engine(
        "postgresql+psycopg2://postgres:1234@localhost:5432/postgres"
        )
        loaded_data = load_data('historical_weather_aqi_table',engine)
        train_data, val_data, test_data = preprocess_data(loaded_data)
        save_data( train_data, val_data, test_data,engine)
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)

if __name__ == '__main__':
    main()




    







