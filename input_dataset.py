import csv
import numpy as np
import random
random.seed(1234)

def read_hr_dataset(test_prop = 0.2):
    with open('hr_dataset.csv', 'rt') as csvfile:
        # import data via csv reader
        spamreader = csv.reader(csvfile)
        index, data = 0, []
        for row in spamreader:
            if index == 0 :
                data_scheme = row
                index += 1
            else:
                data.append(row)        
        # one-hot-encoding for sales department
        departs_list = list({row[-2] for row in data})
        departs_list.sort()
        d2n = dict(zip(departs_list, range(10)))
        salary = {'low':-1, 'medium':0, 'high':1}
        for row in data:
            # convert salary into ordered scale
            row[-1] = salary[row[-1]] 
            # convert department into one-hot-code
            depart = row[-2]
            row += one_hot(d2n[depart])
            # delete original colume of sales department
            del row[8]    
        # Convert from string to numeric data
        data = np.array(data, dtype = float)
        # Random Shuffle the data to randomly select the train set and the test set 
        np.random.shuffle (data)
        print ("Data shape : ", data.shape)
        
        test_size = int (len(data)*test_prop)
        DATA_TEST = data [:test_size]
        DATA_TRAIN = data [test_size:]
        
        # Data Scheme
        data_scheme_x = data_scheme.copy()
        del data_scheme_x [-2]
        del data_scheme_x [6]
        data_scheme_x.extend(departs_list)
        
        data_scheme_y = [data_scheme[6]]
        
        print ("X_SCHEME : ", data_scheme_x)
        print ("Y_SCHEME : ", data_scheme_y)
        
        X_TEST = np.hstack((DATA_TEST[:,:6],DATA_TEST[:,7:]))
        Y_TEST = DATA_TEST[:,6].reshape(-1,1)
        X_TRAIN = np.hstack((DATA_TRAIN[:,:6],DATA_TRAIN[:,7:]))
        Y_TRAIN = DATA_TRAIN[:,6].reshape(-1,1)
        print ("X_TRAIN : ", X_TRAIN.shape)
        print ("Y_TRAIN : ", Y_TRAIN.shape)
        print ("X_TEST : ", X_TEST.shape)
        print ("Y_TEST : ", Y_TEST.shape)
                  
    return X_TRAIN, Y_TRAIN, X_TEST, Y_TEST

def one_hot(num):
    'mapping from integer to one-hot encoding function'
    one_hot = [0 for _ in range(10)]
    one_hot[num] = 1
    return one_hot
