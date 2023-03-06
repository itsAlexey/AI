#Нейронная сеть для предсказания похода на вечеринку с обучнием (без весов)
import numpy
import sys

class PartyNN(object):        
        def __init__(self, learning_rate=0.1):
            self.weights_0_1 = numpy.random.normal(0.0, 2 ** -0.5, (2, 3))
            self.weights_1_2 = numpy.random.normal(0.0, 1, (1, 2))
            self.sigmoid_mapper = numpy.vectorize(self.sigmoid)
            self.learning_rate = numpy.array([learning_rate])
            
        def sigmoid(self, x):
            return 1 / (1 + numpy.exp(-x))
        
        def predict(self, inputs):
            inputs_1 = numpy.dot(self.weights_0_1, inputs)
            outputs_1 = self.sigmoid_mapper(inputs_1)
            
            inputs_2 = numpy.dot(self.weights_1_2, outputs_1)
            outputs_2 = self.sigmoid_mapper(inputs_2)
            return outputs_2
        
        def train(self, inputs, expected_predict):     
            inputs_1 = numpy.dot(self.weights_0_1, inputs)
            outputs_1 = self.sigmoid_mapper(inputs_1)
            
            inputs_2 = numpy.dot(self.weights_1_2, outputs_1)
            outputs_2 = self.sigmoid_mapper(inputs_2)
            actual_predict = outputs_2[0]
            
            error_layer_2 = numpy.array([actual_predict - expected_predict])
            gradient_layer_2 = actual_predict * (1 - actual_predict)
            weights_delta_layer_2 = error_layer_2 * gradient_layer_2  
            self.weights_1_2 -= (numpy.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate
            
            error_layer_1 = weights_delta_layer_2 * self.weights_1_2
            gradient_layer_1 = outputs_1 * (1 - outputs_1)
            weights_delta_layer_1 = error_layer_1 * gradient_layer_1
            self.weights_0_1 -= numpy.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T  * self.learning_rate
#Среднеквадратичное отклонение
def MSE(y, Y):
    return numpy.mean(y-Y)**2

train = [
    ([0,0,0],0),
    ([0,0,1],1),
    ([0,1,0],0),
    ([0,1,1],0),
    ([1,0,0],1),
    ([1,0,1],1),
    ([1,1,0],0),
    ([1,1,1],1),
]

epochs = 4000
learning_rate = 0.007

network = PartyNN(learning_rate=learning_rate)

for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for input_stat, correct_predict in train:
        network.train(numpy.array(input_stat), correct_predict)
        inputs_.append(numpy.array(input_stat))
        correct_predictions.append(numpy.array(correct_predict))
    
    train_loss = MSE(network.predict(numpy.array(inputs_).T), numpy.array(correct_predictions))
    sys.stdout.write("\rProgress: {}, Training loss: {}".format(str(100 * e/float(epochs))[:4], str(train_loss)[:5]))


for input_stat, correct_predict in train:
    print("For input: {} the prediction is: {}, expected: {}".format(
            str(input_stat),
            str(network.predict(numpy.array(input_stat)) > .5),
            str(correct_predict == 1)))