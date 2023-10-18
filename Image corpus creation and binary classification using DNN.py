#!/usr/bin/env python
# coding: utf-8

# # Lab4: Image corpus creation and binary classification using DNN

# Name : Deepa S
# 
# Roll No : 225229107

# In[49]:


import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# In[50]:


barn_owl_folder = "C:/Users/2mscdsa07/Desktop/owl"
similar_images_folder = 'C:/Users/2mscdsa17/Desktop/apple'


# In[51]:


images = []
labels = []

# Read barn owl images
for filename in os.listdir(barn_owl_folder):
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    image_path = os.path.join(barn_owl_folder, filename)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    
    images.append(image)
    labels.append(0)  # Label 0 for barn owls


# In[52]:


# Read similar images
for filename in os.listdir(similar_images_folder):
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    image_path = os.path.join(similar_images_folder, filename)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    
    images.append(image)
    labels.append(1)  # Label 1 for similar images


# In[53]:


# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# In[55]:


# Define the model
model = Sequential()
model.add(Flatten(input_shape=(64, 64, 3)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[56]:


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[57]:


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)


# In[34]:


# Evaluate the model on test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# In[35]:


# Predict class labels for test images
test_predictions = model.predict(X_test)


# In[36]:


# Convert probabilities to class labels (0 or 1)
test_predictions = np.round(test_predictions).flatten()


# In[37]:


# Print the predicted labels and the actual labels
print("Predicted Labels:", test_predictions)
print("Actual Labels:", y_test)


# In[38]:


# Save the image corpus and labels
np.save('image_corpus.npy', image_corpus)
np.save('labels.npy', labels)


# In[39]:


# Load the image corpus and labels
image_corpus = np.load('image_corpus.npy')
labels = np.load('labels.npy')

# Print the shapes and contents of the loaded arrays
print("Image Corpus shape:", image_corpus.shape)
print("Image Corpus:")
print(image_corpus)

print("\nLabels shape:", labels.shape)
print("Labels:")
print(labels)


# In[ ]:




