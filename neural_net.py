import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import math
from mfcc_extract_train import trainx_main
from mfcc_extract_test import testx_main
from output import trainy_main
from output import testy_main
from keras.constraints import maxnorm

trainx = []; trainy = []; testx = []; testy = []

trainx = trainx_main()
testx = testx_main()
trainy = trainy_main()
testy = testy_main()
		
x_train = np.array(trainx)
y_train = np.array(trainy)
x_test = np.array(testx)
y_test = np.array(testy)

model = Sequential()
#model.add(Dropout(0.3), input_shape = (113136,))
model.add(Dense(100, activation='sigmoid', input_dim=113136, kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(36, activation='softmax' , kernel_constraint=maxnorm(3)))

#sgd = SGD(lr=0.01, decay=1e-4, momentum=0.99, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=100,
          batch_size=12)
train_score = model.evaluate(x_train, y_train, batch_size=12)
print train_score

print '----------------Training Complete-----------------'

test_score = model.evaluate(x_test, y_test, batch_size = 12)
print test_score
