import pandas as pd
import os 
import logging
import yaml
from sqlalchemy import create_engine
from sqlalchemy import text

logs_dir = '/Users/amritamandal/Desktop/Python/MLOPS/DVC_AWS/MLOPS-1/logs'
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger('feature_preprocessing_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(logs_dir, 'feature_preprocessing_engineering.log')
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

def create_features(data):
    try:
        data['date'] = pd.to_datetime(data['date'])
        data['week'] = data['date'].dt.isocalendar().week
        data['weekday'] = data['date'].dt.weekday
        data['month'] = data['date'].dt.month
        data['temp_range'] = data['temperature_2m_max'] - data['temperature_2m_min']
        data['app_temp_range'] = data['apparent_temperature_max'] - data['apparent_temperature_min']
        data['photochemical_activity_index'] = data['shortwave_radiation_sum']*((data['temperature_2m_max']+data['temperature_2m_min']/2))

        lags = [7]
        for lag in lags:
            data[f"pm2_5_mean_lag_{lag}"] = (
                data
                .groupby(['state','place','latitude','longitude'])["pm2_5_mean"]
                .shift(lag)
            )
        for window in [7,14]:
            data[f"pm25_rolling_mean_{window}"] = (
            data
            .groupby(['state','place','latitude','longitude'])["pm2_5_mean"]
            .rolling(window=window)
            .mean()
            .reset_index(level=[0,1], drop=True)
            )
        data = data.dropna()
        data['radiation_rate'] = data['shortwave_radiation_sum']/(data['sunshine_duration']+0.001)
        data['uv_index_temp'] = data['apparent_temperature_max']*data['uv_index_max']
        data['ozone_temp_max'] = data['ozone_mean']* data['temperature_2m_max']
        data['ozone_uv_mean'] = data['ozone_mean']* data['uv_index_mean']
        data['ozone_sunshine'] = data['ozone_max'] * data['sunshine_duration']
        data['dust_wind_speed'] = data['dust_mean']*data['wind_speed_10m_max']
        data.drop(columns = ['state', 'place', 'latitude', 'longitude', 'date','pm2_5_mean'],inplace=True)
        logger.debug("Data has been pre-processed and features created")
        return data 
    
    except Exception as e:
        logger.debug(f"Error has occured - {e}")
        raise

def save_data(processed_data,split,engine):
    try:
        processed_data.to_sql(
        name=f"weather_aqi_processed_{split}",
        con=engine,
        schema="public",        
        if_exists="replace",     
        index=False
        )
        logger.debug(f"Processed data has been saved into the database  -> weather_aqi_processed_{split}")
    
    except Exception as e:
        logger.debug(f"Error has occured - {e}")
        raise

def main():
    try:
        engine = create_engine(
        "postgresql+psycopg2://postgres:1234@localhost:5432/postgres"
        )
        train_data = load_data("weather_aqi_train", engine)
        val_data = load_data("weather_aqi_val", engine)
        test_data = load_data("weather_aqi_test", engine)

        train_data_processed = create_features(train_data)
        val_data_processed = create_features(val_data)
        test_data_processed = create_features(test_data)

        save_data(train_data_processed,'train',engine)
        save_data(val_data_processed,'val',engine)
        save_data(test_data_processed,'test',engine)
    
    except Exception as e:
        logger.debug(f"Error has occured - {e}")
        raise

if __name__ == '__main__':
    main()



    
