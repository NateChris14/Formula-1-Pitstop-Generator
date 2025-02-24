import os
import sys
from dataclasses import dataclass

from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(os.getcwd(),'artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
            "Logistic Regression": LogisticRegression(),
            "K-Neighbors Classifier": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "XGBClassifier": XGBClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "Support Vector Classifier": SVC()
            }

            params = {
                    "Logistic Regression":{
                    'C': [0.01, 0.1, 1],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [100],
                    'class_weight': [None, 'balanced'],
                    'l1_ratio': [0.1, 0.5]  # Only used when penalty='elasticnet'
                },

                "K-Neighbors Classifier":{

                    'n_neighbors': [1, 3, 5, 10],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski'],
                    'algorithm': ['auto', 'ball_tree']
                },

                "Decision Tree":{

                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [3, 5, 10, 20, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': [None, 'sqrt', 'log2'],
                    'splitter': ['best', 'random']
                },

                "Random Forest":
                {
                    'n_estimators': [50, 100],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [5, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [2, 5],
                    'max_features': ['sqrt', None],
                    'bootstrap': [True, False]
                },

                "XGBClassifier":{
                    'n_estimators': [100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.6, 0.8],
                    'colsample_bytree': [0.8, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [0.01, 0.1, 1]
                },

                "AdaBoostClassifier":{
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'algorithm': ['SAMME', 'SAMME.R'],
                    'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)]
                },

                "Support Vector Classifier":{
                    'C': [0.01, 0.1, 1],
                    'kernel': ['linear', 'poly', 'rbf'],
                    'degree': [2, 3],
                    'gamma': ['scale', 'auto'],
                    'coef0': [0.0, 0.1]
                }

            }

            model_report, best_model = evaluate_models(
                X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                models=models,param=params
            )

            #Ensuring best model and model_report are valid
            if not model_report or best_model is None:
                raise CustomException("No best model found")
            
            best_model_score = max(model_report.values(), default=None)

            #Ensuring the best model score is assigned
            if best_model_score is None or best_model_score < 0.6:
                raise CustomException("No suitable model found")
            
            logging.info(f"Best model with precision score : {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            precision = precision_score(y_test,predicted)

            return precision
        
        except Exception as e:
            raise CustomException(e,sys)