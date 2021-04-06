import sys
from RandomForest import *

def do_action(action):
    
    k_value = 0
    if(action == 'Validate'):
        print('K value: ', end='')
        k_value = int(input())

    print('Path to training dataset: ', end='')
    training_dataset = input()

    test_dataset = ''
    if(action == 'Classify'):
        print('Path to test dataset: ', end='')
        test_dataset = input()

    is_numeric = ''
    while(is_numeric != 'y' and is_numeric != 'n'):
        print('Is data type numeric (y/n): ', end='')
        is_numeric = input()
        is_numeric = 'n' if is_numeric == '' else is_numeric

    print('Dataset separator: ', end='')
    sep = input()

    print('Target column name: ', end='')
    target_column_name = input()
    
    print('Number of trees: ', end='')
    number_of_trees = int(input())
    
    print('Bootstrap size (Default: Dataset size):', end='')
    bootstrap_size = input()

    should_print_tree = ''
    while(should_print_tree != 'y' and should_print_tree != 'n'):
        print('Should print tree (y/n, default: n): ', end='')
        should_print_tree = input()
        should_print_tree = 'n' if should_print_tree == '' else should_print_tree

    vary_tree = ''
    while(vary_tree != 'y' and vary_tree != 'n'):
        print('Vary tree (y/n, default: y): ', end='')
        vary_tree = input()
        vary_tree = 'y' if vary_tree == '' else vary_tree
        
    print('Running...')

    RF = RandomForest()
    RFDataset = pd.read_csv(training_dataset, sep=sep, engine='python')
    RFPredictive = list(RFDataset.columns)
    RFPredictive.remove(target_column_name)
    listOfTrees = RF.train(RFDataset, RFPredictive, target_column_name, number_of_trees, len(RFDataset) if bootstrap_size == '' else int(bootstrap_size), (True if should_print_tree=='y' else False), (True if vary_tree=='y' else False), (True if is_numeric=='y' else False))

    if(action == 'Classify'):
        RFDatasetTest = pd.read_csv(test_dataset, sep=sep, engine='python')
        listOfPredictions = RF.predict(listOfTrees, RFDatasetTest)
        RF.voting(listOfPredictions, RFDatasetTest)

    if(action == 'Validate'):
        folds = kFoldSplit(k_value, RFDataset, target_column_name)
        RF.crossValidation(k_value, folds, RFPredictive, target_column_name, number_of_trees, len(RFDataset) if bootstrap_size == '' else int(bootstrap_size), (True if should_print_tree=='y' else False),(True if vary_tree=='y' else False))


if(len(sys.argv) != 2):
    print('Call python3 Application.py [action]')
    print('Possible actions:')
    print('\t- Training')
    print('\t- Classify')
    print('\t- Validate')
    exit()

try:
    if(sys.argv[1] == 'Training'):
        print('Selected option: Treino')
        do_action('Training')
    else:
        if(sys.argv[1] == 'Classify'):
            print('Selected option: Classificar')
            do_action('Classify')
        else:
            if(sys.argv[1] == 'Validate'):
                print('Selected option: Validação')
                do_action('Validate')
            else:
                print('Ação Inválida!')
                print('Possible actions:')
                print('\t- Training')
                print('\t- Classify')
                print('\t- Validate')
except:
    print('An error occurred during execution. Check the input information.')