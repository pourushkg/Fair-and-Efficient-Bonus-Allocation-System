import os
from src.Bonus_Allocation_System.logging import logger
from src.Bonus_Allocation_System.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        logger.info("Shape of the data")
        logger.info(data.shape)

        logger.info("Information about the data")
        logger.info(data.info())

        logger.info("Type Casting (converting customer_id datatype from int to object)")
        data["customer_id"]=data["customer_id"].astype("object")

        logger.info("Partitioning the dataset based on feature type, separating numerical and categorical attributes.")
        numerical_data = data.select_dtypes(exclude="object")
        numerical_features = data.select_dtypes(exclude="object").columns

        categorical_data = data.select_dtypes(include="object")
        categorical_features = data.select_dtypes(include="object").columns

        logger.info("Shape of the numerical data")
        logger.info(numerical_data.shape)

        logger.info("Shape of the categorical data")
        logger.info(categorical_data.shape)

        logger.info("Information about the numerical data")
        logger.info(numerical_data.info())

        logger.info("Finding the correlation between the numerical data")

        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm',annot_kws={"size":4}) 
        plt.figure(figsize=(30,30))
        plt.savefig("./images/correlation_analysis.png")
        plt.close()


        logger.info("Discription of the numerical data")
        logger.info(numerical_data.describe())

        logger.info("Skewness of the numerical data ")
        logger.info(numerical_data.skew(axis=0,skipna=True))

        logger.info("Kurtosis of the numerical data ")
        logger.info(numerical_data.kurtosis(axis=0,skipna=True))

        logger.info("Handling Duplicates ")
        logger.info(numerical_data.duplicated().sum())
        logger.info("There is no duplicated data present in the dataset")


        logger.info("Finding the outlier in the dataset ")
        numerical_data.plot(kind = 'box', subplots = True,sharey=False, vert =0,sharex = False,layout = [8,2], figsize=(20,20)) 
        plt.subplots_adjust(wspace = 0.75) 
        plt.savefig("./images/outlier.png")
        plt.close()

        logger.info("Applying winsorizer technique in the numerical data")

        
        winsor = Winsorizer(capping_method = 'iqr', 
                          tail = 'both',
                          fold = 1.5,
                          variables = ['age', 'income_level', 'Winning_percentage', 'Days_Since_Last_Bet',
       'Active_Days', 'Total_Number_of_Bets', 'Total_Amount_Wagered',
       'Average_Bet_Amount', 'Number_of_Bonuses_Received',
       'Amount_of_Bonuses_Received', 'Revenue_from_Bonuses',
       'Increase_in_Bets_After_Bonus', 'Increase_in_wagering_after_Bonus',
       'Should_Receive_Bonus'])
        
        numerical_data_1 = pd.DataFrame(winsor.fit_transform(numerical_data), columns = numerical_data.columns).convert_dtypes()

        logger.info("Numerical data after the winsorisation is applied")

        numerical_data.plot(kind = 'box', subplots = True,sharey=False, vert =0,sharex = False,layout = [8,2], figsize=(20,20))
        plt.subplots_adjust(wspace = 0.75)
        plt.savefig("./images/After_cleaning_outlier.png")
        plt.close()

        logger.info("Applying univariate analysis for numerical featues ")

        plt.figure(figsize=(15,15))
        plt.suptitle('Univariate Analysis for Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)


        for i in range(0, len(numerical_features)):
            plt.subplot(8, 2, i+1)
            sns.kdeplot(x=numerical_data[numerical_features[i]], color='blue')
            plt.xlabel(numerical_features[i])
            plt.tight_layout()
        
        plt.savefig("./images/univariate_analysis.png")
        plt.close()


        

        percentage = numerical_data["Should_Receive_Bonus"].value_counts(normalize=True)
        labels = ["Receive","Denied"]
        logger.info("pie chat to show the number of denied and receive bonuses")
        fig, ax = plt.subplots(figsize =(15, 8))
        explode = (0, 0.1)
        colors = ['#1188ff','#e63a2a']
        ax.pie(percentage, labels = labels, startangle = 90,
        autopct='%1.2f%%',explode=explode, shadow=True, colors=colors)
        plt.savefig("./images/Taget_variable_pie_chart.png")
        plt.close()





        X = data[['Winning_percentage', 'Average_Bet_Amount',
       'Number_of_Bonuses_Received', 'Amount_of_Bonuses_Received',
       'Revenue_from_Bonuses','Should_Receive_Bonus']]
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(X)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info("Shape of the training data")
        logger.info(train.shape)
        logger.info("Shape of the test data")
        logger.info(test.shape)
