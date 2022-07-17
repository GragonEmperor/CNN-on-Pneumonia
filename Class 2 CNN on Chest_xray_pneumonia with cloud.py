#!/usr/bin/env python
# coding: utf-8

# In[1]:


train_root  = "E:/Files/2022_NTU/Week_2/Class_2/data/Chest_xray_pneumonia/train"
test_root = "E:/Files/2022_NTU/Week_2/Class_2/data/Chest_xray_pneumonia/test"
print(train_root)
print(test_root)


# In[2]:


batch_size = 2


# In[3]:


from keras.preprocessing.image import ImageDataGenerator

Generator = ImageDataGenerator()
train_data = Generator.flow_from_directory(train_root, (150, 150), batch_size=batch_size)
test_data = Generator.flow_from_directory(test_root, (150, 150), batch_size=batch_size)


# In[4]:


# show the image
from skimage import io
image = io.imread("E:/Files/2022_NTU/Week_2/Class_2/data/Chest_xray_pneumonia/test/NORMAL/IM-0001-0001.jpeg")
# the size of the image
print(image.shape)
io.imshow(image)


# In[5]:


# make sure the dataset is inside the program
import tensorflow as tf
from matplotlib.pyplot import imshow
import os

im = train_data[0][0][1]
img = tf.keras.preprocessing.image.array_to_img(im)
imshow(img)

num_classes = len([i for i in os.listdir(train_root)])
print(num_classes)


# In[6]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.05))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.05))

model.add(Dense(num_classes, activation="softmax"))
model.summary()


# In[7]:


#remove optimizer if needed
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(train_data, batch_size = batch_size, epochs=2)


# In[8]:


# score = model.evaluate(train_data)
# print(score)
score = model.evaluate(test_data)
print(score)


# In[9]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[10]:


plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[11]:


dir(model); #检测model里面有哪些语法


# In[12]:


import seaborn as sns
import numpy as np
pred = np.argmax(model.predict(test_data),axis=1)
# predict_class，在tensorflow的2.9版本被删了


# In[13]:


pred


# In[14]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_data.classes, pred)
sns.heatmap(cm, annot=True)


# In[15]:


cm


# In[16]:


#depends on number of classes
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))


# In[17]:


sum(sum(cm))


# In[18]:


cm[0,0]+cm[1,1]


# In[19]:


#import joblib
#joblib.dump(model,"pneumonia")

from keras.models import save_model
save_model(model, "Pneumonia")


# In[ ]:


from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from PIL import Image #use PIL
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def init():
    if request.method == 'POST':
        file = request.files['file']
        print("File Received")
        filename = secure_filename(file.filename)
        print(filename)
        # Open the image form working directory
        image = Image.open(file)
        model = load_model("Pneumonia")
        img = np.asarray(image)
        img.resize((150,150,3))
        img = np.asarray(img, dtype="float32") #need to transfer to np to reshape
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) #rgb to reshape to 1,100,100,3
        pred=model.predict(img)
        return(render_template("index.html", result=str(pred)))
    else:
        return(render_template("index.html", result="WAITING"))
if __name__ == "__main__":
    app.run()


# In[ ]:





# In[ ]:




