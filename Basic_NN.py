import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('heart.csv')
df.head() # For cross-checking whether the data is loaded or not

#Separating the independent and dependent features
x = np.array(df.loc[ :, df.columns != 'output'])
y = np.array(df['output'])

#Shapes of x and y
print(f"x: {x.shape}, y: {y.shape}")

#Train, Test split
#x_train: The features used for training the model.
#x_test: The features used for evaluating the trained model's performance.
#y_train: The corresponding target labels for the training data.
#y_test: The corresponding target labels for the testing data.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123) #"random_state" ensures that the data is split in the same way every time you run your code, which can be useful for reproducibility

#Scaling the data
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

class NeuralNetwork:
    def __init__(self, LR, x_train, x_test, y_train, y_test):
        self.w = np.random.randn(x_train.shape[1]) #weihts
        self.b = np.random.randn() #bias
        self.LR = LR #(LR): "learning rate" controls the size of the steps taken during parameter updates
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.L_train = []
        self.L_test = []
        
    def activation(self, x):
        return 1 / (1 + np.exp(-x)) #sigmoid function

    def deactivation(self,x):
        # derivative of sigmoid
        return self.activation(x) * (1 - self.activation(x)) # σ(x)*(1 − σ(x))
    
    def forward(self, x):
        hidden_layer_1 = np.dot(x, self.w) + self.b
        activation_1 = self.activation(hidden_layer_1)
        return activation_1
    
    def backward(self, x, y_true):
        # calc gradients
        hidden_layer_1 = np.dot(x, self.w) + self.b
        y_pred = self.forward(x)
        dL_dpred = 2 * (y_pred - y_true)
        dpred_dhidden1 = self.deactivation(hidden_layer_1)
        dhidden1_db = 1
        dhidden1_dw = x
        
        dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db
        dL_dw = dL_dpred * dpred_dhidden1 * dhidden1_dw
        
        return dL_db, dL_dw
    
    def optimizer(self, dL_db, dL_dw): 
        #Optimizer: responsible for updating the model's parameters ('b' and 'w') based on the calculated gradients (dL_db and dL_dw) using a specified learning rate (LR).
        self.b = self.b - dL_db * self.LR
        self.w = self.w - dL_dw * self.LR
        
    def train(self, ITERATIONS):
        for i in range(ITERATIONS):
            # random position
            random_pos = np.random.randint(len(self.x_train))
            
            # forward pass
            y_train_true = self.y_train[random_pos]
            y_train_pred = self.forward(self.x_train[random_pos])
            
            # calc training loss
            L = np.sum(np.square(y_train_pred - y_train_true))
            self.L_train.append(L)
            
            # calc gradients
            dL_db, dL_dw = self.backward(
                self.x_train[random_pos], self.y_train[random_pos]
            )
            # update weights
            self.optimizer(dL_db, dL_dw)

            # calc error at every epoch end
            L_sum = 0
            for j in range(len(self.x_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.x_test[j])
                L_sum += np.square(y_pred - y_true)
            self.L_test.append(L_sum)

        return "training successfully finished" 
        
# Hyper parameters
LR = 0.1
ITERATIONS = 1000

# model instance and training
nn = NeuralNetwork(LR=LR, x_train=x_train_scaler, y_train=y_train, x_test=x_test_scaler, y_test=y_test)
nn.train(ITERATIONS=ITERATIONS)
    
# iterate over test data
total = x_test_scaler.shape[0]
correct = 0
y_preds = []
for i in range(total):
    y_true = y_test[i]
    y_pred = np.round(nn.forward(x_test_scaler[i]))
    y_preds.append(y_pred)
    correct += 1 if y_true == y_pred else 0
    
# Calculate Accuracy
acc = correct / total

# Baseline Classifier
from collections import Counter
Counter(y_test)

# Confusion Matrix
confusion_matrix(y_true = y_test, y_pred = y_preds)

