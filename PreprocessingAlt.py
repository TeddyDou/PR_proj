'''
Created on Nov 29, 2017

@author: Zac
'''
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class PreprocessAlt():
    
    def __init__(self, data_path):
        self.data = pd.read_excel(data_path).as_matrix()
        self.matrix_default = np.empty([6636, 24], dtype=float)
        self.matrix_nondefault = np.empty([6636, 24], dtype=float)
        self.matrix_x_train = np.empty([10616, 23], dtype=float)
        self.matrix_y_train = np.empty([10616, ], dtype=float)
        self.matrix_x_test = np.empty([2656, 23], dtype=float)
        self.matrix_y_test = np.empty([2656, ], dtype=float)
        self.x_train = np.empty([10616, 7], dtype=float)
#         self.y_train = np.empty([10616, ], dtype=int)
        self.x_test = np.empty([2656, 7], dtype=float)
#         self.y_test = np.empty([2656, ], dtype=int)
#         data_x_train, data_x_test, data_y_train, data_y_test
#         print(self.data)
#         print(self.data.shape)
#     
    def data_reduce(self):
        m_nondefault = np.empty([23364, 24], dtype=int)
        #non default index
        i = 0
        #default index
        j = 0
        for row in range(1,30001):
            if self.data.item(row, 24) == 0:
                m_nondefault[i, :] = self.data[row, 1:]
                i += 1
            else:
                self.matrix_default[j,:] = self.data[row, 1:]
                j += 1
#         print("default :%d"%j)
#         print(self.matrix_default)
#         print(self.matrix_default.shape)

        #reduce data into 13272 records with 6636 default and 6636 non-default 
        matrix_reduced_nondefault = random.sample(list(m_nondefault), 6636)
        m = 0
        for array in matrix_reduced_nondefault:
            self.matrix_nondefault[m, :] = array
            m += 1   
#         print("non default: %d"%m)
#         print(self.matrix_nondefault)
#         print(self.matrix_nondefault.shape)
     
    def set_education(self):    
        #outlier handling: set the education 0,4,5,6 to others 4
        for row in range(0,6636):
            if self.matrix_nondefault.item(row,2) in [0, 4, 5, 6]:
                self.matrix_nondefault[row, 2] = 4
            if self.matrix_default.item(row,2) in [0, 4, 5, 6]:
                self.matrix_default[row, 2] = 4
    
    def set_marriage(self):    
        #outlier handling: set the education 0,3 to others 3 
        for row in range(0,6636):
            if self.matrix_nondefault.item(row,3) in [0, 3]:
                self.matrix_nondefault[row, 3] = 3
            if self.matrix_default.item(row,3) in [0, 3]:
                self.matrix_default[row, 3] = 3
    
    def data_divide(self):
        
        X_nondefault_datax = self.matrix_nondefault[:, :-1]
        X_nondefault_datay = self.matrix_nondefault[:, 23]
#         print("non")
#         print(X_nondefault_datax)
#         print(X_nondefault_datay)
        X_default_datax = self.matrix_default[:, :-1]
        X_default_datay = self.matrix_default[:, 23]
#         print("de")
#         print(X_default_datax)
#         print(X_default_datay)
#         
        X_nondefault_train, X_nondefault_test, Y_nondefault_train, Y_nondefault_test = train_test_split(X_nondefault_datax, X_nondefault_datay, test_size=0.2)
        X_default_train, X_default_test, Y_default_train, Y_default_test = train_test_split(X_default_datax, X_default_datay, test_size=0.2)
        
      
        #traning data
        data_x_train = np.concatenate((X_nondefault_train, X_default_train), axis=0)
#         print(X_nondefault_train)
#         print (X_default_train)
#         print(data_x_train)
#         print(data_x_train.shape)
#         
        data_y_train = np.concatenate((Y_nondefault_train, Y_default_train), axis=0)
#         print(Y_nondefault_train)
#         print (Y_default_train)
#         print(data_y_train)
#         print(data_y_train.shape)
#         
        #testing data
        data_x_test= np.concatenate((X_nondefault_test, X_default_test), axis=0)
#         print(X_nondefault_test)
#         print (X_default_test)
#         print(data_x_test.shape)
#         
        data_y_test = np.concatenate((Y_nondefault_test, Y_default_test), axis=0)
#         print(Y_nondefault_test)
#         print (Y_default_test)
#         print(data_y_test.shape)
#         
        self.matrix_x_train = data_x_train
#        print(self.matrix_x_train)
        self.matrix_x_test = data_x_test
#        print(self.matrix_x_test)
        self.matrix_y_train = data_y_train
#         print(self.matrix_y_train.shape)
        self.matrix_y_test = data_y_test
#         print(self.matrix_y_test.shape)
#         
        return data_x_train, data_x_test, data_y_train, data_y_test
    
    def set_missed_payments(self):
        #generate missed payments based on PAY_1 through PAY_6
        self.x_train[:, 0:5] = self.matrix_x_train[:, 0:5]
        self.x_test[:, 0:5] = self.matrix_x_test[:, 0:5]
#         print(self.matrix_x_train[1])
#         print(self.x_train[1])
#         print(self.matrix_x_test[1])
#         print(self.x_test[1])
        
        
        
        for row in range(0,10616):
            missed_payment = self.matrix_x_train[row, 5:11].max()
            #if the missed payment is -2, set it to -1
            if missed_payment == -2:
                self.x_train[row, 5] = -1
            else:
                self.x_train[row, 5] = missed_payment
                
        
        for row in range(0,2656):
            missed_payment = self.matrix_x_test[row, 5:11].max()
            #if the missed payment is -2, set it to -1
            if missed_payment == -2:
                self.x_test[row, 5] = -1
            else:
                self.x_test[row, 5] = missed_payment
    
    def set_amt_owed(self):
        #sum up BILL_AMT1 through BILL_AMT6, sum up PAY_AMT1 through PAY_AMT6, generate the AMT_OWED by subtraction 
        for row in range(0,10616):
            self.x_train[row, 6] = self.matrix_x_train[row, 11:17].sum() - self.matrix_x_train[row, 17:23].sum()
        for row in range(0,2656):
            self.x_test[row, 6] = self.matrix_x_test[row, 11:17].sum() - self.matrix_x_test[row, 17:23].sum() 
        
#         print("*******")              
#         print(self.matrix_x_train[1])
#         print(self.x_train[1])
#         print("*********")
#         print(self.matrix_x_test[1])
#         print(self.x_test[1]) 
#         print(self.matrix_x_train[2])
#         print(self.x_train[2])
#         print(self.matrix_x_test[2])
#         print(self.x_test[2]) 
        
        
#        print(self.x_train)
#        print(self.x_test)
  
        
        return self.x_train, self.x_test, self.matrix_y_train, self.matrix_y_test
    
    
    def load(self):
        self.data_reduce()
        self.set_education()
        self.set_marriage()
        return self.data_divide()
     
    
    def dimension_decrease(self):
        self.set_missed_payments()
        return self.set_amt_owed()
    
        

if __name__ == '__main__':
    a = PreprocessAlt("default of credit card clients.xls")
    a.load()
    a.dimension_decrease()
