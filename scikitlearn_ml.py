
import config
from scikitlearn_class import ScikitLearnClass
 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,  cross_val_score, cross_val_predict
from sklearn import metrics

def main():
#      class object of the ann library
    scikit_learn_class = ScikitLearnClass()
    
#      read csv data file
    df_name = scikit_learn_class.read_data(config.FILE_NAME, config.ENCODING_LATIN1)
    
#      show file information
    scikit_learn_class.show_file_information(df_name)
    
#      calculate descriptive statistics
    scikit_learn_class.show_descriptive_statistics(df_name)
    
#      select X label and Y target data frames
    X_label, Y_target = scikit_learn_class.select_label_target(df_name, config.TARGET_COLUMN_NAME)
    
#    data split in train and test (this should be model evaluation data, no test data)
    X_label_train, X_label_test, Y_target_train, Y_target_test = scikit_learn_class. train_test_split_data(X_label, Y_target, config.PERCENT_TEST_SIZE, config.RANDOM_STATE)
    
#     set data standard scaler
    X_label_train_scaled, X_label_test_scaled = scikit_learn_class.scaler_data(X_label_train, X_label_test, config.DATA_STANDARD_SCALER)
    
#     model training
    mlp_classifier = scikit_learn_class.ann_training_model(X_label_train_scaled, Y_target_train, config.HIDDEN_LAYER_NEURON_SIZES, config.ACTIVATION_FUNCTION, config.SOLVER_OPTIMIZATION, config.MAXIMUM_ITERATION, config.RANDOM_STATE)
    
#     model prediction
    Y_target_predicted = scikit_learn_class.predict_model(mlp_classifier, X_label_test_scaled)
    
#     model evaluation     
    scikit_learn_class.classification_evaluate_model(Y_target_test, Y_target_predicted)
         
if __name__ == '__main__':
    main()