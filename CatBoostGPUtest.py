import time
from loguru import logger
from datasets import load_dataset
from catboost import CatBoostClassifier, Pool


dataset = load_dataset("mstz/uscensus")
train_data = dataset['train'].to_pandas()
X = train_data.drop(columns=['dYrsserv'])
y = train_data['dYrsserv']
train_pool = Pool(X, y)

model = CatBoostClassifier(
    iterations=1000,
    task_type="GPU",
    devices='0:1',
    verbose=False,
)

for i in range(20):
    start_time = time.time()
    model.fit(train_pool)
    end_time = time.time()
    logger.info(f"Training of Catboost completed in {(end_time - start_time):.6f} seconds")
