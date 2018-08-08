
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from scikitlearn_superclass import ScikitLearnSuperClass

class ScikitLearnClass(ScikitLearnSuperClass):

    def __init__(self):
        super().__init__()
                    
    def ann_training_model(self, X_label_train_scaled, Y_label_train, hidden_layer_neuron_sizes, activation_function, solver_optimization, maximum_iteration, random_state):        
        '''
        create and fit the multi-layer perceptron classifier
        :param X_label_train_scaled: x label train scaled
        :param Y_label_train: y label train
        :param hidden_layer_neuron_sizes: hidden layer neuron sizes
        :param activation_function: activation function
        :param solver_optimization: solver optimization
        :param maximum_iteration: maximum iteration
        :param random_state: random state instance value
        :return multi-layer perceptron classifier (model)
        '''
        try:
            mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_neuron_sizes, activation=activation_function, solver=solver_optimization, max_iter=maximum_iteration, random_state=random_state)
            mlp_classifier.fit(X_label_train_scaled, Y_label_train)
        except Exception:
            self.print_exception_message()
        return mlp_classifier    
    
    def rf_training_model(self, X_label_train_scaled, Y_label_train, number_of_trees, split_criterion, max_number_features, tree_max_depth, random_state):      
        """
       create and fit the random forest classifier
        :param X_label_train_scaled: x label train scaled
        :param Y_label_train: y label train
        :param number_of_trees: number of trees in the forest
        :param split_criterion: measure the quality of a split
        :param max_number_features: number of features to consider when looking for the best split
        :param tree_max_depth: maximum depth of the tree
        :param random_state: random state instance value
        """
        try:
            rf_classifier = RandomForestClassifier(n_estimators=number_of_trees, criterion=split_criterion, max_features=max_number_features, max_depth=tree_max_depth, random_state=random_state)
            rf_classifier.fit(X_label_train_scaled, Y_label_train)
        except Exception:
            self.print_exception_message()
        return rf_classifier  
    
    
    
    