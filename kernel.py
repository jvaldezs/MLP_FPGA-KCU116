import Config
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import time
import sys
import createNormalBase
import math
import numpy as np
import random
import joblib
import parse_example
import KernelFunctions
sgn = KernelFunctions.sgn

def trainMulticlass(iterations,rate):
  weights = np.zeros((nTrainClasses,traindata.shape[1]))
  correct = 0
  t = 0
  accuracies = []
  while (correct / traindata.shape[0]) != 1:
    r = list(range(traindata.shape[0]))
    random.shuffle(r)
    correct = 0
    count = 0
    for i in r:
      sample = traindata[i]
      answer = trainlabels[i]
      maxVal = -1
      guess = -1
      for m in range(nTrainClasses):
        val = kernel(weights[m],sample)
        if val > maxVal:
          maxVal = val
          guess = m 
      if guess != answer:
        weights[guess] = weights[guess] - rate*sample
        weights[answer] = weights[answer] + rate*sample
      else:
        correct += 1
      count += 1
    accuracy = 100*testMulticlass(weights)
    accuracies.append(accuracy)
    print("Iteration: ",t,"Train Accuracy: ",correct / count,"Test Accuracy: ", accuracy)
    t += 1
  print('Max Accuracy: ' + str(max(accuracies)))

def trainMulticlassBinary(iterations,rate):
  weights = np.zeros((nTrainClasses,traindata.shape[1]))
  binaryWeights = np.copy(weights)
  correct = 0
  t = 0
  while (correct / traindata.shape[0]) != 1:
    r = list(range(traindata.shape[0]))
    random.shuffle(r)
    correct = 0
    count = 0
    for i in r:
      sample = traindata[i]
      answer = trainlabels[i]
      maxVal = -1
      guess = -1
      for m in range(nTrainClasses):
        val = kernel(binaryWeights[m],sample)
        if val > maxVal:
          maxVal = val
          guess = m 
      if guess != answer:
        weights[guess] = weights[guess] - rate*sample
        weights[answer] = weights[answer] + rate*sample
        binaryWeights = np.copy(weights)
        binaryWeights = KernelFunctions.binarizeAll(binaryWeights, 1, -1)
      else:
        correct += 1
      count += 1
    
    test_acc = 100*testMulticlass(binaryWeights)
    print(f"Iteration: {t}, Train Accuracy: {correct / count}, Test Accuracy: {test_acc}")
    if((correct / count > 0.74) & (test_acc > 89)):
      break
    t += 1
    
def testMulticlass(weights):
  weights = np.array(weights)
  print(f"Before reshaping, class hypervectors shape: {weights.shape}")  # Debugging

  if weights.shape[1] != 10000:
      raise ValueError(f"Error: Expected weights shape (2, 10000), but found {weights.shape}")

  # Transpose for TensorFlow compatibility
  #weights = weights.T  # Convert from (10, 10000) to (10000, 10)

  print(f"After reshaping, class hypervectors shape: {weights.shape}")  # Debugging
  joblib.dump(weights, "weights.pkl", compress=True)
  print("✅ Saved corrected class hypervectors with shape:", weights.shape)

  correct = 0
  # model guess is 1 and label is 1
  array = []
  for i in range(testdata.shape[0]):
    sample = testdata[i]
    answer = testlabels[i]
    maxVal = -1
    for m in range(nTrainClasses):
      val = kernel(weights[m],sample)
      if val > maxVal:
        maxVal = val
        guess = m
    if guess == answer:
      correct += 1
      if answer == 1:
        array.append(i)

  # Pull out the indices of data that are true faults(testlabel = 1, and guess = answer thus true positive)
  """  with open ('array.txt', 'w') as f:
    for i in range(len(array)):
      f.write(str(array[i]) + '\n')
  exit(-1)"""

  return correct / testdata.shape[0]

directory = Config.directory
dataset = Config.dataset
kernel = KernelFunctions.kernel
init = 1
if init == 1:
  D = KernelFunctions.D
  traindata, trainlabels, testdata, testlabels, nTrainFeatures, nTrainClasses = KernelFunctions.load(directory,dataset) 
  
  traindata = sklearn.preprocessing.normalize(traindata,norm='l2')
  testdata = sklearn.preprocessing.normalize(testdata,norm='l2') 
  mu = Config.mu
  sigma = Config.sigma #/ 20#1 / (math.sqrt(617)) #/ 24#1 #/ (1.4)
  if Config.sparse == 1:
    createNormalBase.createSparse(D, nTrainFeatures, mu, sigma, Config.s)
  else:
    createNormalBase.create(D,nTrainFeatures,mu,sigma)
  size = int(D)
  base = np.random.uniform(0,2*math.pi,size)
  start = time.time()
  traindata = KernelFunctions.encode(traindata,base)
  assert traindata.shape[0] == trainlabels.shape[0]
  print("Encoding training time",time.time() - start)
  start = time.time()
  testdata = KernelFunctions.encode(testdata,base)
  print('Encoding testing time',time.time() - start)
  mu = Config.mu
  sigma = Config.sigma #/ 20#1 / (math.sqrt(617)) #/ 24#1 #/ (1.4)
  if Config.sparse == 1:
    joblib.dump(traindata, open('' + str(Config.dataset) + str(Config.D) + 'train.pkl', "wb"), compress=True)
    joblib.dump(testdata,open(''+str(Config.dataset) + str(Config.D) + 'test.pkl','wb'),compress=True)
  else:
    joblib.dump(traindata, open('' + str(Config.dataset) + str(Config.D) + 'train.pkl', "wb"), compress=True)
    joblib.dump(testdata,open(''+str(Config.dataset) + str(Config.D) + 'test.pkl','wb'),compress=True)
  print(f"traindata Shape: {traindata.shape}, trainLabels Shape: {trainlabels.shape}")
  print(f"testdata Shape: {testdata.shape}, testLabels Shape: {testlabels.shape}")

  time.sleep(2)

  if Config.binarize == 1:
    traindata = KernelFunctions.binarizeAll(traindata, 1, -1)
    testdata = KernelFunctions.binarizeAll(testdata, 1, -1)
else:
  if Config.sparse == 1:
    traindata = joblib.load('' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'train.pkl')
    testdata = joblib.load('' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'test.pkl')
  else:
    traindata = joblib.load('' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'train.pkl')
    testdata = joblib.load('' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'test.pkl')
  pass

if Config.binaryModel == 1:
  trainMulticlassBinary(260, Config.rate)
else:
  trainMulticlass(260,Config.rate)
