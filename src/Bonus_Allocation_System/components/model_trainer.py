import pandas as pd
import os
from src.Bonus_Allocation_System.logging import logger
from sklearn.neighbors import KNeighborsClassifier
import joblib
from src.Bonus_Allocation_System.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        train_data.head()


        train_x = train_data[['Winning_percentage', 'Average_Bet_Amount',
       'Number_of_Bonuses_Received', 'Amount_of_Bonuses_Received',
       'Revenue_from_Bonuses']]
        test_x = test_data[['Winning_percentage', 'Average_Bet_Amount',
       'Number_of_Bonuses_Received', 'Amount_of_Bonuses_Received',
       'Revenue_from_Bonuses']]
        train_y = train_data[['Should_Receive_Bonus']]
        test_y = test_data[['Should_Receive_Bonus']]


        lr = KNeighborsClassifier()
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))