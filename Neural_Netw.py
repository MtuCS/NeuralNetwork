# pip install tensorflow <-- cài thư viện này trước
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

# load the dataset
dataset = pd.read_csv('Silver_data.csv')
print(dataset.shape)

X1 = dataset['n-2'].values
X2 = dataset['n-1'].values
X3 = dataset['n'].values
X4 = dataset['n+1'].values
Y = dataset['n+2'].values

XX = np.column_stack([X1, X2, X3, X4])

print(XX.shape)
print(Y.shape)

#tạo mô hình AI có 3 lớp
model = Sequential()
model.add(Dense(150, input_shape = (4,), activation='tanh'))     
model.add(Dense(150, activation ='tanh'))                      
model.add(Dense(100, activation ='tanh'))                        
model.add(Dense(1, activation = "linear"))  # 1 lớp ra                

print(model.summary())  

# compile the keras model
model.compile(loss='mse', optimizer='Adamax')                    

# fit the keras model on the dataset
 #train 1000 lần
model.fit(XX, Y, epochs=1000, batch_size=128)                    

Y1 = model.predict(XX) #Dữ liệu Y1 dự báo từ mô hình (dự đoán ngày n+2)
print(Y1)

df = pd.DataFrame(Y1, columns=['Predicted_Output'])
df.to_csv('predicted_output.csv', index=False)

fig, ax = plt.subplots()
ax.plot(Y, 'b')
ax.plot(Y1, 'r')
plt.show()

model.save('my_model.keras')