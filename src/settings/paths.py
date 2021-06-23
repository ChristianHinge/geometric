import os

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir,os.path.pardir))

MODELS_PATH = os.path.join(MODULE_PATH, 'models')
MODEL_STORE_PATH = os.path.join(MODULE_PATH, 'model_store')
CHECKPOINT_PATH = os.path.join(MODELS_PATH,'checkpoints')
DATA_PATH = os.path.join(MODULE_PATH, 'data')
CLEANED_DATA_PATH = os.path.join(DATA_PATH, 'cleaned')
NOT_CLEANED_DATA_PATH = os.path.join(DATA_PATH, 'not_cleaned')
VISUALIZATION_PATH = os.path.join(MODULE_PATH, 'reports', 'figures')
CLOUD_PATH = os.path.join(MODULE_PATH,'src','cloud')
