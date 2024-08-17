import pandas as pd
import os
from src.Bonus_Allocation_System.logging import logger
import joblib
from src.Bonus_Allocation_System.entity.config_entity import ModelTrainerConfig
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


        
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        train_data.head()
        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            "K-Neighbors Classifier": KNeighborsClassifier(),
            "XGBClassifier": XGBClassifier(), 
            "CatBoosting Classifier": CatBoostClassifier(verbose=False),
            "Support Vector Classifier": SVC(),
            "AdaBoost Classifier": AdaBoostClassifier()
            }
        def evaluate_clf(true, predicted):
            acc = accuracy_score(true, predicted) # Calculate Accuracy
            f1 = f1_score(true, predicted) # Calculate F1-score
            precision = precision_score(true, predicted) # Calculate Precision
            recall = recall_score(true, predicted)  # Calculate Recall
            roc_auc = roc_auc_score(true, predicted) #Calculate Roc
            return acc, f1 , precision, recall, roc_auc

        X_train = train_data[['Winning_percentage', 'Average_Bet_Amount',
       'Number_of_Bonuses_Received', 'Amount_of_Bonuses_Received',
       'Revenue_from_Bonuses']]
        X_test = test_data[['Winning_percentage', 'Average_Bet_Amount',
       'Number_of_Bonuses_Received', 'Amount_of_Bonuses_Received',
       'Revenue_from_Bonuses']]
        y_train = train_data[['Should_Receive_Bonus']]
        y_test = test_data[['Should_Receive_Bonus']]
        models_list = []
        accuracy_list = []
        auc= []
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Training set performance
            model_train_accuracy, model_train_f1,model_train_precision,\
            model_train_recall,model_train_rocauc_score=evaluate_clf(y_train ,y_train_pred)


            # Test set performance
            model_test_accuracy,model_test_f1,model_test_precision,\
            model_test_recall,model_test_rocauc_score=evaluate_clf(y_test, y_test_pred)

            print(list(models.keys())[i])
            models_list.append(list(models.keys())[i])

            print('Model performance for Training set')
            print("- Accuracy: {:.4f}".format(model_train_accuracy))
            print('- F1 score: {:.4f}'.format(model_train_f1)) 
            print('- Precision: {:.4f}'.format(model_train_precision))
            print('- Recall: {:.4f}'.format(model_train_recall))
            print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))

            print('----------------------------------')

            print('Model performance for Test set')
            print('- Accuracy: {:.4f}'.format(model_test_accuracy))
            accuracy_list.append(model_test_accuracy)
            print('- F1 score: {:.4f}'.format(model_test_f1))
            print('- Precision: {:.4f}'.format(model_test_precision))
            print('- Recall: {:.4f}'.format(model_test_recall))
            print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))
            auc.append(model_test_rocauc_score)
            print('='*35)
            print('\n')
        
        report=pd.DataFrame(list(zip(models_list, accuracy_list)), columns=['Model Name', 'Accuracy']).sort_values(by=['Accuracy'], ascending=False)
        logger.info("Final accurary table")
        print(report.to_string(index=False))
        xbg = XGBClassifier()
        joblib.dump(xbg, os.path.join(self.config.root_dir, self.config.model_name))
        return report

    


     #   joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))