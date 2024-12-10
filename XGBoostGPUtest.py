import time
import xgboost as xgb
from loguru import logger
from datasets import load_dataset


dataset = load_dataset("mstz/uscensus")
train_data = dataset['train'].to_pandas()
X = train_data.drop(columns=['dYrsserv'])
y = train_data['dYrsserv']
dtrain = xgb.DMatrix(X, label=y)

params = {
    'objective': 'multi:softmax',
    'eval_metric': 'merror',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'learning_rate': 0.1,
    'max_depth': 10,
    'num_class': 3,
    'n_estimators': 1000,
}

logger.info("Training started")
for i in range(20):
    start_time = time.time()
    model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])
    end_time = time.time()
    logger.info(f"Training of XGBoost completed in {(end_time - start_time):.6f} seconds")
