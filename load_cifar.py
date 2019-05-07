import pickle
import numpy as np
import math

#Step 1: define a function to load traing batch data from directory
def load_training_batch(folder_path,batch_id):
    with open(folder_path + "data_batch_" + str(batch_id), 'rb') as fo:
        train_dict = pickle.load(fo, encoding = 'latin1')
    ###fetch features using the key ['data']###
    features = train_dict['data']
    ###fetch labels using the key ['labels']###
    labels = train_dict['labels']
    return features,labels

#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):

    ###load batch using pickle###
    with open(folder_path + "test_batch", 'rb') as fo:
        test_dict = pickle.load(fo, encoding = 'latin1')
    ###fetch features using the key ['data']###
    features = test_dict['data']
    ###fetch labels using the key ['labels']###
    labels = test_dict['labels']
    return features,labels

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names():
    label_List = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    return label_List

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
    """
    Args:
    features: a numpy array with shape (10000, 3072)
    Return:
    features: a numpy array with shape (10000,32,32,3)
    """
    shaped_Features = np.reshape(features, (10000,32,32,3))
    return shaped_Features

#Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path,batch_id,data_id):
    pass

#Step 6: define a function that does min-max normalization on input
def normalize(x):
    return (x - x.min())/(x.max() - x.min())

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
    NUM_CLASSES = 10
    one_hot_encoding_matrix = np.zeros([len(x), NUM_CLASSES])
    #return np.eye(x.size)[x]
    for i in range(0, len(x)):
        one_hot_encoding_matrix[i, x[i]] = 1
        
    return one_hot_encoding_matrix

#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features,labels,filename):
    normal_feature = normalize(features)
    one_hot_labels = one_hot_encoding(labels)
    
    preprocessed_dict = {'normal feature': normal_feature, 'one hot labels': one_hot_labels}
    with open(filename, 'wb') as handle:
        pickle.dump(preprocessed_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)  #Make sure about this operation

#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
    raw_train_data_feature1, raw_train_data_label1 = load_training_batch(folder_path,1)
    raw_train_data_feature2, raw_train_data_label2 = load_training_batch(folder_path,2)
    raw_train_data_feature3, raw_train_data_label3 = load_training_batch(folder_path,3)
    raw_train_data_feature4, raw_train_data_label4 = load_training_batch(folder_path,4)
    raw_train_data_feature5, raw_train_data_label5 = load_training_batch(folder_path,5)
    
    raw_test_data_feature, raw_test_data_label = load_testing_batch(folder_path)
    
    #build the validation data: 10% of 50000 = 5000
    raw_validate_data_features = raw_train_data_feature1[0:5000, :]
    raw_validate_data_label = raw_train_data_label1[0:5000]
    preprocess_and_save(raw_train_data_feature1, raw_train_data_label1, 'processed_train1')
    preprocess_and_save(raw_train_data_feature2, raw_train_data_label2, 'processed_train2')
    preprocess_and_save(raw_train_data_feature3, raw_train_data_label3, 'processed_train3')
    preprocess_and_save(raw_train_data_feature4, raw_train_data_label4, 'processed_train4')
    preprocess_and_save(raw_train_data_feature5, raw_train_data_label5, 'processed_train5')
    
    preprocess_and_save(raw_test_data_feature, raw_test_data_label, 'processed_test')
    
    preprocess_and_save(raw_validate_data_features, raw_validate_data_label, 'processed_validate')

#Step 10: define a function to yield mini_batch
def mini_batch(features,labels,mini_batch_size):
    #build the validation data: 10% of 50000 = 5000
    #raw_validate_data_features = raw_train_data_feature1[0:5000, :]
    #raw_validate_data_label = raw_train_data_label1[0:5000]
    
    total_size = features.shape[0]
    num_mini_batches = math.floor(total_size/mini_batch_size) 
    for k in range(0, num_mini_batches):
        mini_features = features[k*mini_batch_size : (k + 1)*mini_batch_size, :]
        mini_labels   = labels[k*mini_batch_size : (k + 1)*mini_batch_size, :]
        #return all of the mini batches:
        yield mini_features, mini_labels

#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id,mini_batch_size):
    file_name = 'processed_train'
    with open(file_name + str(batch_id), 'rb') as fo:
        preprocessed_train_dict = pickle.load(fo, encoding = 'latin1')
    ###fetch features using the key ['data']###
    features = preprocessed_train_dict['normal feature']
    labels   = preprocessed_train_dict['one hot labels']
    #features, labels = load_training_batch(file_name, mini_batch_size)
    return mini_batch(features,labels,mini_batch_size)

#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch():
    file_name = 'processed_validate'
    #features,labels = 
    
    with open(file_name, 'rb') as fo:
        preprocessed_train_dict = pickle.load(fo, encoding = 'latin1')
    ###fetch features using the key ['data']###
    features = preprocessed_train_dict['normal feature']
    labels   = preprocessed_train_dict['one hot labels']
    
    return features,labels

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
    file_name = 'processed_test'
    with open(file_name, 'rb') as fo:
        preprocessed_train_dict = pickle.load(fo, encoding = 'latin1')
    features = preprocessed_train_dict['normal feature']
    labels   = preprocessed_train_dict['one hot labels']
    
    return mini_batch(features,labels,test_mini_batch_size)

