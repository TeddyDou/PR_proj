'''
Created on Nov 29, 2017

@author: Zac
'''

from PreprocessingAlt import PreprocessAlt

class Postprocess():
    
    def __init__(self, x1, x2, y1, y2):
        self.x_train = x1
        self.x_test = x2
        self.y_train = y1
        self.y_test = y2
        # print("#######")
        # print(x1)
        # print(x1.shape)
        # print("*******")
        # print(x2)
        # print(x2.shape)
        # print("*******")
    
    def set_age(self):
        #group age into different intervals
        for row in range(0, 10616):
            if self.x_train[row, 4] in range(1,10):
                self.x_train[row, 4] = 1
            elif self.x_train[row, 4] in range(10,20):
                self.x_train[row, 4] = 2
            elif self.x_train[row, 4] in range(20,30):
                self.x_train[row, 4] = 3
            elif self.x_train[row, 4] in range(30,40):
                self.x_train[row, 4] = 4    
            elif self.x_train[row, 4] in range(40,50):
                self.x_train[row, 4] = 5
            elif self.x_train[row, 4] in range(50,60):
                self.x_train[row, 4] = 6
            elif self.x_train[row, 4] in range(60,70):
                self.x_train[row, 4] = 7
            elif self.x_train[row, 4] in range(70,80):
                self.x_train[row, 4] = 8
            else:
                self.x_train[row, 4] = 9
            print("aged row: %d" % row)
                
        for row in range(0, 2656):
            if self.x_test[row, 4] in range(1,10):
                self.x_test[row, 4] = 1
            elif self.x_test[row, 4] in range(10,20):
                self.x_test[row, 4] = 2
            elif self.x_test[row, 4] in range(20,30):
                self.x_test[row, 4] = 3
            elif self.x_test[row, 4] in range(30,40):
                self.x_test[row, 4] = 4    
            elif self.x_test[row, 4] in range(40,50):
                self.x_test[row, 4] = 5
            elif self.x_test[row, 4] in range(50,60):
                self.x_test[row, 4] = 6
            elif self.x_test[row, 4] in range(60,70):
                self.x_test[row, 4] = 7
            elif self.x_test[row, 4] in range(70,80):
                self.x_test[row, 4] = 8
            else:
                self.x_test[row, 4] = 9   
                
    def set_amount(self):
        
        #process training data
        for row in range(0, 10616):
         
            #group credit limit into different intervals
            if self.x_train[row, 0] == 0:
                continue
            elif self.x_train[row, 0] in range(1,100001):
                self.x_train[row, 0] = 1
            elif self.x_train[row, 0] in range(100001,500001):
                self.x_train[row, 0] = 2
            else:
                self.x_train[row, 0] = 3
            
         
            #group owed amount into different intervals
            if self.x_train[row, 6] in range(-100000,0):
                self.x_train[row, 6] = -1
            elif self.x_train[row, 6] in range(-500000,-100000):
                self.x_train[row, 6] = -2
            elif self.x_train[row, 6] < -500000:
                self.x_train[row, 6] = -3       
            elif self.x_train[row, 6] == 0:
                continue
            elif self.x_train[row, 6] in range(1,100001):
                self.x_train[row, 6] = 1
            elif self.x_train[row, 6] in range(100001,500001):
                self.x_train[row, 6] = 2
            else:
                self.x_train[row, 6] = 3

            print("amount row: %d" % row)

    

        #process testing data
        for row in range(0, 2656):
           
            #group credit limit into different intervals
            if self.x_test[row, 0] == 0:
                continue
            elif self.x_test[row, 0] in range(1,100001):
                self.x_test[row, 0] = 1
            elif self.x_test[row, 0] in range(100001,500001):
                self.x_test[row, 0] = 2
            else:
                self.x_test[row, 0] = 3
            
            #group owed amount into different intervals
            if self.x_test[row, 6] in range(-100000,0):
                self.x_test[row, 6] = -1
            elif self.x_test[row, 6] in range(-500000,-100000):
                self.x_test[row, 6] = -2
            elif self.x_test[row, 6] < -500000:
                self.x_test[row, 6] = -3       
            elif self.x_test[row, 6] == 0:
                continue
            elif self.x_test[row, 6] in range(1,100001):
                self.x_test[row, 6] = 1
            elif self.x_test[row, 6] in range(100001,500001):
                self.x_test[row, 6] = 2
            else:
                self.x_test[row, 6] = 3        
        
        # print(self.x_train)
        # print("*******")
        # print(self.x_test)
                
    def improve_data(self):
        self.set_age()
        self.set_amount()
        return self.x_train, self.x_test, self.y_train, self.y_test
        
if __name__ == '__main__':
    import pickle
    a = PreprocessAlt("default of credit card clients.xls")
    rx1, rx2, ry1, ry2 = a.load()
    x1, x2, y1, y2 = a.dimension_decrease()
    b = Postprocess(x1,x2,y1,y2)
    xd1, xd2, yd1, yd2= b.improve_data()
    with open("express_x", "w") as f:
        pickle.dump(rx1, f)
        pickle.dump(rx2, f)
        pickle.dump(x1, f)
        pickle.dump(x2, f)
        pickle.dump(xd1, f)
        pickle.dump(xd2, f)
        f.close()
    with open("express_y", "w") as fp:
        pickle.dump(ry1, fp)
        pickle.dump(ry2, fp)
        pickle.dump(y1, fp)
        pickle.dump(y2, fp)
        pickle.dump(yd1, fp)
        pickle.dump(yd2,fp)
        f.close()