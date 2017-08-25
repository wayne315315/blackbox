
# coding: utf-8

# In[1]:


"""
Author contributed : LIN HSUAN YU
This script is developed and tested under the following environment:
- Tensorflow-GPU : 1.3.0
- Keras : 2.0.7 
- Python : 3.5.2
- CUDA SDK : 8.0
- cudnn : 5.1
"""
import time
t0 = t1 = time.time()
from input_dataset import read_hr_dataset
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
import random, h5py
t2 = time.time()
random.seed(1234567890)
print ("Import time :", t2-t1, "seconds")


# In[2]:


t1 = time.time()
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = read_hr_dataset()
t2 = time.time()
print ("Data reading time :", t2-t1, "seconds")


# In[3]:


# Set keras learning phase and backend
keras.layers.core.K.set_learning_phase(1)
config = tf.ConfigProto(device_count = {'GPU' : 0})
sess = tf.InteractiveSession(config=config)
keras.backend.set_session(sess)


# In[4]:


# Construct the oracle with a keras model : 3 FC-layer
t1 = time.time()
oracle = Sequential()
oracle.add(Dense(1000, input_dim = 18))
oracle.add(Activation('relu'))
oracle.add(Dense(100))
oracle.add(Activation('sigmoid'))
oracle.add(Dense(1))
oracle.add(Activation('sigmoid'))

# Compile the oracle model for binary classification
oracle.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

t2 = time.time()
print ("Oracle Construction and Compilation time :", t2-t1, "seconds")


# In[5]:


# Train the oracle model (epoch = 1000)
t1 = time.time()
oracle.fit(X_TRAIN, Y_TRAIN, epochs=1000, batch_size=64, verbose=0)
t2 = time.time()
print ("Oracle Training time :", t2-t1, "seconds")


# In[6]:


# Saved oracle configuration and weights to disk with json and hdf respectively
t1 = time.time()
print("Saved oracle configuration to disk")
oracle_json = oracle.to_json()
with open("model/oracle.json", "w") as oracle_file:
    oracle_file.write(oracle_json)

print("Saved oracle weights to disk")
oracle.save_weights("model/oracle_weights.h5")
t2 = time.time()
print ("Oracle saving time :", t2-t1, "seconds")


# In[7]:


# Output of the train set from the oracle model
output_train = oracle.predict(X_TRAIN, batch_size=12000, verbose=0)
output_train = np.array([1 if num >= 0.5 else 0 for num in output_train]).reshape(-1,1)
correct_list = np.array([1 if output == y else 0 for output,y in zip(output_train,Y_TRAIN)])
print("Accuracy of the oracle on the train set :", sum (correct_list)/12000)


# In[8]:


# Construct a substitute model with 4-layer FC model via Keras
t1 = time.time()
sub = Sequential()
sub.add(Dense(200, input_dim = 18))
sub.add(Activation('sigmoid'))
sub.add(Dense(1))
sub.add(Activation('linear'))
t2 = time.time()
print ("Substitute Construction time :", t2-t1, "seconds")


# In[9]:


# Select initial training samples from original test set for the substitute (sample_size=1000)
t1, sample_size = time.time(), 1000
while True :
    sub_sample_x = np.array(random.sample (list(X_TEST), sample_size))
    output_sample = oracle.predict(sub_sample_x, batch_size = sample_size, verbose = 0)
    output_sample = np.array([1.0 if num >= 0.5 else 0.0 for num in output_sample]).reshape(-1,1)
    if (sum(output_sample) > 0.2*sample_size and sum(output_sample) < 0.8*sample_size) or time.time()-t1 > 10 :
        break
t2 = time.time()
print ("Sample Generation time :", t2-t1, "seconds")
print (sum(output_sample))


# In[10]:


# Train the substitute with increased samples generated from Jacobian-based Augmentation

# Determine the training parameters
cycles, batch_size, epochs, lmbda, learning_rate = 5, 128, 10000, 0.1, 0.005

# Only X0~X4 are continuous variables and therefore adjustable
x = tf.placeholder(tf.float32, shape=[None, 18])
y = tf.placeholder(tf.float32, shape=[None,1])
y_pred = sub(x)
prob = tf.sigmoid(y_pred)

# Define the loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = y_pred))

# Define the training optimizer
train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
sess.run(tf.global_variables_initializer())

# Reload oracle weights due to previous initializer
oracle.load_weights("model/oracle_weights.h5")

# Begin training
t1 = time.time()
with sess.as_default():
    start, length = 0, len(output_sample)
    permut = np.random.permutation(range(length))
    batch_x, batch_y = sub_sample_x[permut], output_sample[permut]
    grads = tf.gradients(prob, x) # 1-dim Jacobian matrix

    for cycle in range(cycles):
        length, t3 = len(batch_y), time.time()
        print("====================== Cycle %d started. ======================"%(cycle))
        print("Batch size :", length)
        
        # Begin training (epochs = 10000)
        for epoch in range(epochs):
            start = 0
            permut = np.random.permutation(range(length))
            batch_x, batch_y = batch_x[permut], batch_y[permut]
            while True:
                end = start + (batch_size - 1)
                if end >= length:
                    break
                next_batch_x, next_batch_y = batch_x[start:end], batch_y[start:end]
                train_step.run(feed_dict={x:next_batch_x, y:next_batch_y})
                start += batch_size
            if epoch % 1000 == 0:
                t5 = time.time()
            elif (epoch+1) % 1000 == 0:
                print("loss :", sess.run(loss, feed_dict={x:batch_x, y:batch_y}))
                print("******** Epoch %d took %d seconds. ********"%(epoch+1, time.time()-t5))

        # Jacobian_based augmentation (lambda = 0.1), only adjust X0 ~ X4
        if cycle < cycles -1 :
            augmented_batch_x, t4 = [], time.time()
            for batch in batch_x :
                grad = sess.run(tf.sign(grads), feed_dict={x:batch.reshape(1,-1)}).reshape(-1,) ##
                augmented_batch_x.append(batch + lmbda * np.hstack((grad[:5],np.zeros(13))))
            print ("##### Cycle %d : %d seconds for sample generation. #####" %(cycle, time.time()-t4))
            
            augmented_batch_x = np.array(augmented_batch_x)
            augmented_batch_y = oracle.predict(augmented_batch_x, batch_size=length, verbose=1)
            augmented_batch_y = np.array([1 if num >= 0.5 else 0 for num in augmented_batch_y]).reshape(-1,1)
            batch_x = np.vstack((batch_x, augmented_batch_x))
            batch_y = np.vstack((batch_y, augmented_batch_y))
        print("# Real loss :", sess.run(loss, feed_dict={x:X_TRAIN, y:output_train}))
        print("====================== Cycle %d took %d seconds. ======================"%(cycle, time.time()-t3))

t2 = time.time()
print("====================== Substitute Training time :", t2-t1, "seconds ======================")


# In[11]:


# Saved substitute configuration and weights to disk with json and hdf respectively
t1 = time.time()
print("Saved substitute configuration to disk")
sub_json = sub.to_json()
with open("model/sub.json", "w") as sub_file:
    sub_file.write(sub_json)

print("Saved substitute weights to disk")
sub.save_weights("model/sub_weights.h5")
t2 = time.time()
print ("Substitute saving time :", t2-t1, "seconds")


# In[12]:


# Test : Oracle
output_test = oracle.predict(X_TEST, batch_size=2999, verbose=0)
output_test = np.array([1 if num >= 0.5 else 0 for num in output_test])
correct_list = np.array([1 if output == y else 0 for output,y in zip(output_test,Y_TEST)])
print("Output of the oracle on the test set :", output_test)
print("Accuracy of the oracle on the test set :", sum (correct_list)/2999)


# In[13]:


# Test : Substitute
prob = tf.sigmoid(sub(x))
result = sess.run(prob, feed_dict={x:X_TRAIN})
result = [1 if num > 0.5 else 0 for num in result]
output = oracle.predict(X_TRAIN, batch_size=12000, verbose=0)
output = np.array([1 if num >= 0.5 else 0 for num in output])
correct_list = np.array([1 if output == y else 0 for output,y in zip(result, output)])
print("Accuracy of the substitute on the prediction of oracle output :",sum(correct_list)/12000)
correct_list = np.array([1 if output == y else 0 for output,y in zip(result, Y_TRAIN)])
print("Accuracy of the substitute on the prediction of real output :",sum(correct_list)/12000)


# In[14]:


# Initiate a blackbox attack with adversarial examples generating 
# from substitute model by Fast Gradient Sign Method (Default setting : L1 norm)

# Define loss function of substitute model and calculate its gradients with respect to x
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = y_pred))
grads = tf.gradients(loss, x)[0][0]

# Set the parameter epsilon = 0.5 for FGSM (Only alter the first 5 terms , X0~X4)
print ("Fast Gradient Sign Method based on L1 norm")
t1 = time.time()
epsilon, adv_X_TEST, adv_Y_TEST = 0.5, [], Y_TEST
for i in range(len(X_TEST)):
    grads_val = sess.run(grads, feed_dict={x:X_TEST[i].reshape(1,-1), y:Y_TEST[i].reshape(1,-1)})
    grads_val = np.hstack((grads_val[:5], np.zeros(13)))
    grads_l_1 = grads_val/sum(abs(grads_val)) if sum(abs(grads_val)) != 0 else np.zeros(18)
    grads_l_inf = sess.run(tf.sign(grads_val))
    
    aug = grads_l_1 # set aug = grads_l_inf if L-infinity norm is adopted 
    adv_X_TEST.append(X_TEST[i] + epsilon * aug)

adv_X_TEST = np.array(adv_X_TEST)
print ("Adversarial samples generation took ", time.time()-t1, "seconds")


# In[15]:


# Evaluate the accuarcy of the adversarial samples on the oracle
output_test = oracle.predict(adv_X_TEST, batch_size=2999, verbose=0)
output_test = np.array([1 if num >= 0.5 else 0 for num in output_test])
correct_list = np.array([1 if output == y else 0 for output,y in zip(output_test,adv_Y_TEST)])
print("Output of the oracle on the adversarial samples :", output_test)
print("Accuracy of the oracle on the adversarial samples :", sum (correct_list)/2999, "\n")


# In[16]:


# Evaluate the accuarcy of the random augmented samples on the oracle
rand_X_TEST, rand_Y_TEST = [], Y_TEST
for i in range(len(X_TEST)):
    rands = np.random.random_sample((5,))
    symbols = np.array([-1.0,1.0])
    unit, signs = rands/sum(rands), np.random.choice(symbols, 5)
    aug_rand = np.hstack(([num * sign for num, sign in zip (unit,signs)], np.zeros(13)))
    rand_X_TEST.append(X_TEST[i] + epsilon * aug_rand)
rand_X_TEST = np.array(rand_X_TEST)

output_test = oracle.predict(rand_X_TEST, batch_size=2999, verbose=0)
output_test = np.array([1 if num >= 0.5 else 0 for num in output_test])
correct_list = np.array([1 if output == y else 0 for output,y in zip(output_test,rand_Y_TEST)])
print("Output of the oracle on the random augmented samples :", output_test)
print("Accuracy of the oracle on the random augmented samples :", sum (correct_list)/2999, "\n")


# In[17]:


print ("Total execution time :", time.time()-t0, "seconds")

