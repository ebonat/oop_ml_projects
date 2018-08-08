
import os
import sys
import traceback
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import config

from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ScikitLearnSuperClass(object):

    def __init__(self):
        pass
       
    def read_data(self, csv_file_name, encoding_unicode=None):
        '''
        read data from a csv file
        :param file_name: csv file path and name    
        :param encoding_unicode: csv file encoding unicode
        :return data frame
        '''
        df_file_name = None
        try:
            project_directory_path = os.path.dirname(sys.argv[0])  
            csv_file_path_name = os.path.join(project_directory_path, csv_file_name)  
            if encoding_unicode is None:
                df_file_name = pd.read_csv(filepath_or_buffer=csv_file_path_name, sep=",")   
            else:
                df_file_name = pd.read_csv(filepath_or_buffer=csv_file_path_name, sep=",", encoding=encoding_unicode)               
        except Exception:
            self.print_exception_message()
        return df_file_name
    
    def show_file_information(self, df_name):
        '''
        show data file information
        :param df_name: data frame
        :return none
        '''
        try:           
            print("DATASET INFORMATION")
            df_name.info()
            print()
        except Exception:
            self.print_exception_message()
                       
    def show_file_data(self, df_name):
        '''
        print data file
        :param df_name: data frame
        :return none
        '''
        try:          
            print("DATASET")
            print(df_name)
            print()          
        except Exception:
            self.print_exception_message()
        
    def show_descriptive_statistics(self, df_name):
        '''
        show descriptive statistics for numerical labels (features)
        :param df_name: data frame
        :return none
        '''
        try:
            print("DESCRIPTIVE STATISTICS")     
            descriptive_statistics = df_name.describe().transpose()           
            print(descriptive_statistics)  
            print()        
        except Exception:
            self.print_exception_message()
            
    def select_label_target(self, df_name, target_column):
        '''
        select x label and y target data frames by target column
        :param df_name: data frame
        :param target_column: target column name
        :return x label and y target data frames
        '''
        try:
            X_label = df_name.drop(labels=target_column, axis=1)
            y_target = df_name[target_column]            
        except Exception:
            self.print_exception_message()
        return X_label, y_target
    
    def train_test_split_data(self, X_label, y_target, test_size_percentage, random_state, stratify_target=None):       
        '''
        select x and y train and test data
        :param X_label: x label data frame
        :param Y_target: y label data frame
        :param test_size_percentage: test size in percentage (%)
        :param random_state: random state initial value
        :param stratify_target: used stratify target (for classification in general)
        return: x and y train and test data
        '''
        try:
            if stratify_target is None:
                X_label_train, X_label_test, y_target_train, y_target_test = train_test_split(X_label, y_target, test_size=test_size_percentage/100, random_state=random_state)
            else:
                X_label_train, X_label_test, y_target_train, y_target_test = train_test_split(X_label, y_target, test_size=test_size_percentage/100, stratify=y_target, random_state=random_state)
        except Exception:
            self.print_exception_message()
        return X_label_train, X_label_test, y_target_train, y_target_test
    
    def scaler_data(self, X_label_train, X_label_test, scaler_type):
        """
        select x label train and test scaled
        :param X_label_train: X label train
        :param X_label_test: X label test
        :param scaler_type: scaler type
        :return x label train and test scaled
        """
        try:
            if (scaler_type == config.DATA_STANDARD_SCALER):
                scaler = StandardScaler()
            elif (scaler_type == config.DATA_ROBUST_SCALER):
                scaler = RobustScaler()
            elif (scaler_type == config.DATA_ROBUST_SCALER):
                scaler = Normalizer()
            elif (scaler_type == config.DATA_ROBUST_SCALER):
                scaler = MinMaxScaler()
            elif (scaler_type == config.DATA_ROBUST_SCALER):
                scaler = MaxAbsScaler()
            scaler.fit(X_label_train)
            X_label_train_scaled = scaler.transform(X_label_train)
            X_label_test_scaled = scaler.transform(X_label_test)
        except Exception:
            self.print_exception_message()
        return X_label_train_scaled, X_label_test_scaled    
    
    def predict_model(self, ml_model, X_label_test_scaled):
        '''
        select y target predicted data frame
        :param ml_model: machine learning model
        :param X_label_test_scaled: x label test scaled
        :return y target predicted
        '''
        try:
            y_target_predicted = ml_model.predict(X_label_test_scaled)
        except Exception:
            self.print_exception_message()
        return y_target_predicted
    
    
    def classification_evaluate_model(self, y_target_test, y_target_predicted):
        '''
        print confusion matrix, classification report and accuracy score values
        :param y_target_test: y target test
        :param y_target_predicted: y target predicted
        :return none
        '''
        try:
            confusion_matrix_value = confusion_matrix(y_target_test, y_target_predicted)
            print("CONFUSION MATRIX")
            print(confusion_matrix_value)
            print()
            
            classification_report_result = classification_report(y_target_test, y_target_predicted)        
            print('CLASSIFICATION REPORT')
            print(classification_report_result)
            print()        
    
            accuracy_score_value = accuracy_score(y_target_test, y_target_predicted) * 100
            accuracy_score_value = float("{0:.2f}".format(accuracy_score_value))
            print("ACCURACY SCORE")        
            print( "{} %".format(accuracy_score_value))
            print()
        except Exception:
            self.print_exception_message()    
    
    def print_exception_message(self, message_orientation="horizontal"):
        """
        print full exception message
        :param message_orientation: horizontal or vertical
        :return none
        """
        try:
            exc_type, exc_value, exc_tb = sys.exc_info()            
            file_name, line_number, procedure_name, line_code = traceback.extract_tb(exc_tb)[-1]       
            time_stamp = " [Time Stamp]: " + str(time.strftime("%Y-%m-%d %I:%M:%S %p")) 
            file_name = " [File Name]: " + str(file_name)
            procedure_name = " [Procedure Name]: " + str(procedure_name)
            error_message = " [Error Message]: " + str(exc_value)        
            error_type = " [Error Type]: " + str(exc_type)                    
            line_number = " [Line Number]: " + str(line_number)                
            line_code = " [Line Code]: " + str(line_code) 
            if (message_orientation == "horizontal"):
                print( "An error occurred:{};{};{};{};{};{};{}".format(time_stamp, file_name, procedure_name, error_message, error_type, line_number, line_code))
            elif (message_orientation == "vertical"):
                print( "An error occurred:\n{}\n{}\n{}\n{}\n{}\n{}\n{}".format(time_stamp, file_name, procedure_name, error_message, error_type, line_number, line_code))
            else:
                pass                    
        except Exception:
            pass