import os

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

MODELS_PATH = os.path.join(MODULE_PATH, 'models')
CHECKPOINT_PATH = os.path.join(MODELS_PATH,'checkpoints')
DATA_PATH = os.path.join(MODULE_PATH, 'data')
CLEANED_DATA_PATH = os.path.join(DATA_PATH, 'cleaned')
NOTCLEANED_DATA_PATH = os.path.join(DATA_PATH, 'not_cleaned')
VISUALIZATION_PATH = os.path.join(MODULE_PATH, 'reports', 'figures')
CLOUD_PATH = os.path.join(MODULE_PATH,'src','cloud')
