
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
from input_dataset import read_hr_dataset
t0 = t1 = time.time()
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


# Set keras learning phase and backend
keras.layers.core.K.set_learning_phase(1)
config = tf.ConfigProto(device_count = {'GPU' : 0})
sess = tf.InteractiveSession(config=config)
keras.backend.set_session(sess)


# In[3]:


# Load data
t1 = time.time()
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = read_hr_dataset()
t2 = time.time()
print ("Data reading time :", t2-t1, "seconds")


# In[4]:


# Create new oracle and substitute which are identical to the original ones
print ("Loading the oracle and the substitute")
with open("model/oracle.json", "r") as oracle_file:
    new_oracle_json = oracle_file.read()
with open("model/sub.json", "r") as sub_file:
    new_sub_json = sub_file.read()

new_oracle = model_from_json(new_oracle_json)
new_oracle.load_weights("model/oracle_weights.h5")

new_sub = model_from_json(new_sub_json)
new_sub.load_weights("model/sub_weights.h5")


# In[5]:


# Test new oracle (PASSED)
output_test = new_oracle.predict(X_TEST, batch_size=2999, verbose=0)
output_test = np.array([1 if num >= 0.5 else 0 for num in output_test])
correct_list = np.array([1 if output == y else 0 for output,y in zip(output_test,Y_TEST)])
print("Output of the oracle on the test set :", output_test)
print("Accuracy of the oracle on the test set :", sum (correct_list)/2999, "\n")

# Test new substitute (PASSED)
x = tf.placeholder(tf.float32, shape=[None, 18])
y = tf.placeholder(tf.float32, shape=[None,1])
y_pred = new_sub(x)
prob = tf.sigmoid(y_pred)
result = sess.run(prob, feed_dict={x:X_TRAIN})
result = [1 if num > 0.5 else 0 for num in result]
output = new_oracle.predict(X_TRAIN, batch_size=12000, verbose=0)
output = np.array([1 if num >= 0.5 else 0 for num in output])
correct_list = np.array([1 if output == y else 0 for output,y in zip(result, output)])
print("Accuracy of the substitute on the prediction of oracle output :",sum(correct_list)/12000)
correct_list = np.array([1 if output == y else 0 for output,y in zip(result, Y_TRAIN)])
print("Accuracy of the substitute on the prediction of real output :",sum(correct_list)/12000)


# In[6]:


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


# In[7]:


# Evaluate the accuarcy of the adversarial samples on the oracle
output_test = new_oracle.predict(adv_X_TEST, batch_size=2999, verbose=0)
output_test = np.array([1 if num >= 0.5 else 0 for num in output_test])
correct_list = np.array([1 if output == y else 0 for output,y in zip(output_test,adv_Y_TEST)])
print("Output of the oracle on the adversarial samples :", output_test)
print("Accuracy of the oracle on the adversarial samples :", sum (correct_list)/2999, "\n")


# In[8]:


# Evaluate the accuarcy of the random augmented samples on the oracle
rand_X_TEST, rand_Y_TEST = [], Y_TEST
for i in range(len(X_TEST)):
    rands = np.random.random_sample((5,))
    symbols = np.array([-1.0,1.0])
    unit, signs = rands/sum(rands), np.random.choice(symbols, 5)
    aug_rand = np.hstack(([num * sign for num, sign in zip (unit,signs)], np.zeros(13)))
    rand_X_TEST.append(X_TEST[i] + epsilon * aug_rand)
rand_X_TEST = np.array(rand_X_TEST)

output_test = new_oracle.predict(rand_X_TEST, batch_size=2999, verbose=0)
output_test = np.array([1 if num >= 0.5 else 0 for num in output_test])
correct_list = np.array([1 if output == y else 0 for output,y in zip(output_test,rand_Y_TEST)])
print("Output of the oracle on the random augmented samples :", output_test)
print("Accuracy of the oracle on the random augmented samples :", sum (correct_list)/2999, "\n")


# In[9]:


print ("Total execution time :", time.time()-t0, "seconds")

