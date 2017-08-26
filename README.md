# Abstract
The implementation of the black box attack based on https://arxiv.org/abs/1602.02697 <br/>
Human Resource Dataset : https://www.kaggle.com/ludobenistant/hr-analytics

# Declaration
These scripts are partially modified from https://github.com/tensorflow/cleverhans

# Requirement
- Python 3+
- Tensorflow r1.2+
- Keras 2.0+
- git
- numpy, h5py


# How to initiate a black box attack 
Step 1 : Open your terminal and type the following command to download the source code to your current directory:<br/>
```
git clone https://github.com/wayne315315/blackbox.git
```


Step 2 : Change the directory to the "blackbox" folder with the following command:<br/>
```
cd blackbox
(or) 
cd /path/to/blackbox/folder
```
Step 3 : Initiate a black box attack based on the PRE-TRAINED oracle and the substitute model with the following command<br/>
```
python3 hr_adv_only.py
```
Step 4 : (Optional) Initate a black box attack including RE-TRAINING both oracle and substitute model (It may take hours)<br/>
```
python3 hr_blackbox_attack.py
```
Step 5 : (Optinal) If you want to run on GPU, please adjust the session configuration to 'GPU' : 1
```
config = tf.ConfigProto(device_count = {'GPU' : 1})
```
# Introduction
<p>The source code is provided for demonstrating a black box attack on the unknown classification model (oracle) 
for predicting the employment's resignment. </p>
<p>All the architecture and the training data are all invisible to the hacker. 
The only information hackers can get is the data output from the oracle which were fed by the hackers. </p>
<p>In this scenario, hackers can perform a black box attack by constructing a selected model with totally different architecture
and initialization, called the substitue model, which is latter training with the sample set constructed by Jacobian-based augmentation. <br/>
The reason of this step is to simulate the decision boundary of the oracle by training with the data points 
which contributed great variation on the output. </p>
<p>After the substitute model is well-trained, the next step is to generate the adversarial samples via 
Fast gradient sign method (FGSM).
Here we adopt its general form Fast gradient method (FGM) on L1-norm for better performance. </br>
The key to this method is to adding the tiny pertubation on x along the largest gradients of the cost function.
Eventually, the model would assign the wrong label due to the increased cost for assigning the original label.</p>
<p>We also apply a random perturbation of the same L1-norm to the test set as a controlled group.<br/>
Comparing these two kinds of pertubation, you will find that the oracle is pretty robust to the random perturbation,
but extremely vulnerable to the adversarial samples generated from FGM. Therefore, we can safely conclude that 
blind spots of the oracle are diffusely distributed but rare.</p>

<p>Note that in this implementation, we only apply the perturbation on the first five features 
because they are the only five continuous variables in the data set.</p>

<p>For more details, please look up the reference at the abstract.</br>
The original implementation of black box attack on MNIST : https://github.com/tensorflow/cleverhans</p>

# Current setting 
Oracle : 3-layer fully connected neural network (relu - sigmoid - sigmoid) </br>
Substitute : 2-layer fully connected neural network (sigmoid - linear) </br>
Training set size : 12000 </br>
Test set size : 2999 </br>
Cycles for Jacobian-based augmentation (cycles) : 5 </br>
Parameter lambda for Jacobian-based augmentation (lmbda) : 0.1 </br>
Substitute learning rate (learning_rate) : 0.005 </br>
Batch size in substitute learning (batch_size) : 128 </br>
Substitute training epochs during each cycle (epochs) : 10000 </br>
Perturbation in L1-norm during FGM (epsilon) : 0.5 </br>

# Test Environment
- OS : Ubuntu 16.04
- Tensorflow-GPU : 1.3.0
- Keras : 2.0.7 
- Python : 3.5.2
- CUDA SDK : 8.0
- cudnn : 5.1

# Acknowledgement
Author contributed : Lin Hsuan Yu

# Last Note
Please report any runtime error or configuration issue to us, thank you !

# What's next
The high performance version with well-designed data pipeline would be released soon ~
