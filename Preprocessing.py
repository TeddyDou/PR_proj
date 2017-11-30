'''
Created on Nov 25, 2017

@author: Zac
'''

import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class Preprocess():
    
    def __init__(self, data_path):
        self.data = pd.read_excel(data_path).as_matrix()
        print(self.data)
        self.matrix = np.empty([30000,8], dtype=float) 
        self.matrix_standard = np.empty([13272, 8], dtype=float)
#         print(self.matrix)
    
    def set_limit(self):
        self.matrix[0:30000, 0] = self.data[1:30001, 1]
#         print(self.matrix)
    
    def set_sex(self):
        self.matrix[0:30000, 1] = self.data[1:30001, 2]
#         print(self.matrix)
    
    def set_education(self):    
        #set the education 0,4,5,6 to others 4
        for row in range(1,30001):
            if self.data.item(row,3) in [0, 4, 5, 6]:
                self.matrix[row-1, 2] = 4
            else:
                self.matrix[row-1, 2] = self.data[row, 3]
#         print(self.matrix)
    
    def set_marriage(self):    
        #set the education 0,3 to others 3 
        for row in range(1,30001):
            if self.data.item(row,4) in [0, 3]:
                self.matrix[row-1, 3] = 3
            else:
                self.matrix[row-1, 3] = self.data[row, 4]
#         print(self.matrix)
    
    def set_age(self):
        self.matrix[0:30000, 4] = self.data[1:30001, 5]
#         print(self.matrix)
        
    def set_missed_payments(self):
        #generate missed payments based on PAY_1 through PAY_6
        for row in range(1,30001):
            missed_payment = self.data[row, 6:12].max()
            #if the missed payment is -2, set it to -1
            if missed_payment == -2:
                self.matrix[row-1, 5] = -1
            else:
                self.matrix[row-1, 5] = missed_payment
#         print(self.matrix)
        
    def set_amt_owed(self):
        #sum up BILL_AMT1 through BILL_AMT6, sum up PAY_AMT1 through PAY_AMT6, generate the AMT_OWED by subtraction 
        for row in range(1,30001):
            self.matrix[row-1, 6] = self.data[row, 12:18].sum() - self.data[row, 18:24].sum()
    
    def set_default(self):
        self.matrix[0:30000, 7] = self.data[1:30001, 24]
    
    def data_reducation(self):
        matrix_defualt = np.empty([6636, 8], dtype=int)
        matrix_nondefualt = np.empty([23364, 8], dtype=int)
        #non default index
        i = 0
        #default index
        j = 0
        
        for row in range(0,30000):
            if self.matrix.item(row,7) == 0:
                matrix_nondefualt[i,0:8] = self.matrix[row, 0:8]
                i += 1
            else:
                matrix_defualt[j,0:8] = self.matrix[row, 0:8]
                j += 1
        
        #reduce data into 13272 records with 6636 default and 6636 non-default 
        self.matrix_standard[0:6636, 0:8] = matrix_defualt 
        matrix_reduced_nondefault = random.sample(list(matrix_nondefualt), 6636)
        m = 6636
        for array in matrix_reduced_nondefault:
            self.matrix_standard[m, 0:8] = array
            m += 1
        data_x = self.matrix_standard[:, :-1]
        data_y = self.matrix_standard[:, 7]
#         print(self.matrix_standard)
#         print(data_x)
#         print(data_y)
        return data_x, data_y
    
    def data_reducation_alt(self):
        matrix_defualt = np.empty([6636, 8], dtype=float)
        matrix_nondefualt = np.empty([23364, 8], dtype=float)
        #non default index
        i = 0
        #default index
        j = 0
        
        for row in range(0,30000):
            if self.matrix.item(row,7) == 0:
                matrix_nondefualt[i,0:8] = self.matrix[row, 0:8]
                i += 1
            else:
                matrix_defualt[j,0:8] = self.matrix[row, 0:8]
                j += 1
        
        #reduce data into 13272 records with 6636 default and 6636 non-default 
        self.matrix_standard[0:6636, 0:8] = matrix_defualt 
        matrix_reduced_nondefault = random.sample(list(matrix_nondefualt), 6636)
        X_nondefault = np.empty([6636, 8], dtype=float)
        m = 0
        for array in matrix_reduced_nondefault:
            X_nondefault[m, 0:8] = array
            m += 1   
        X_nondefault_datax = X_nondefault[:, :-1]
        X_nondefault_datay = X_nondefault[:, 7]
        X_default_datax = matrix_defualt[:, :-1]
        X_default_datay = matrix_defualt[:, 7]
        
        X_nondefault_train, X_nondefault_test, Y_nondefault_train, Y_nondefault_test = train_test_split(X_nondefault_datax, X_nondefault_datay, test_size=0.2)
        X_default_train, X_default_test, Y_default_train, Y_default_test = train_test_split(X_default_datax, X_default_datay, test_size=0.2)
        
      
        #traning data
        data_x_train = np.concatenate((X_nondefault_train, X_default_train), axis=0)
#         print(X_nondefault_train)
#         print (X_default_train)
#         print(data_x_train)
#         
        data_y_train = np.concatenate((Y_nondefault_train, Y_default_train), axis=0)
#         print(Y_nondefault_train)
#         print (Y_default_train)
#         print(data_y_train)
#         
        #testing data
        data_x_test= np.concatenate((X_nondefault_test, X_default_test), axis=0)
#         print(X_nondefault_test)
#         print (X_default_test)
#         print(data_x_test)
#         
        data_y_test = np.concatenate((Y_nondefault_test, Y_default_test), axis=0)
#         print(Y_nondefault_test)
#         print (Y_default_test)
#         print(data_y_test)
        
        return data_x_train, data_x_test, data_y_train, data_y_test
    


    def load_dataset(self):
        self.set_limit()
        self.set_sex()
        self.set_education()
        self.set_marriage()
        self.set_age()
        self.set_missed_payments()
        self.set_amt_owed()
        self.set_default()
        return self.data_reducation_alt()

if __name__ == '__main__':
    a = Preprocess("default of credit card clients.xls")
    a.load_dataset()

    