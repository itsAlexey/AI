#Нейронная сеть для предсказания похода на вечеринку без обучния (с подобранными весами)
import numpy

alcogol = 1.0
rain = 1.0
friend = 1.0

def activate_function(x):
    if  x >= 0.5:
        return 1
    else:
        return 0

def predict(alcogol, rain, friend):
    inputs = numpy.array([alcogol, rain, friend])
    weights_input_to_hiden_1 = [0.25, 0.25, 0]
    weights_input_to_hiden_2 = [0.5, -0.4, 0.9]
    weights_input_to_hiden = numpy.array([weights_input_to_hiden_1, weights_input_to_hiden_2])
    
    weights_hiden_to_output = numpy.array([-1,1])

    hiden_input = numpy.dot(weights_input_to_hiden, inputs)
    print("hiden input: " + str(hiden_input))
    
    hiden_output = numpy.array([activate_function(x) for x in hiden_input])
    print("hiden output: " + str(hiden_output))

    output = numpy.dot(weights_hiden_to_output, hiden_output)
    print("output: " + str(output))
    return activate_function(output) == 1

print("result: " + str(predict(alcogol, rain, friend)))