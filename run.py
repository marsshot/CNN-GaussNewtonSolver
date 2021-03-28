import numpy
import network as nn

input_size={'n1':8, 'n2':100, 'nc':10}
output_size={'n1':4, 'n2':50, 'nc':1}

def architecture(input_size):
  inp = nn.Input2D(input_size['nc'], input_size['n2'], input_size['n1'])
  layer1 = nn.Conv2D(4,3,3, activation='relu', connection=inp)
  layer2 = nn.Conv2D(2,3,3, activation='relu', connection=layer1)
  layer3 = Merge2D(layer1, layer2)
  layer4 = Pool2D(2,2, mode='max')(layer3)
  layer5 = Conv2D(1,3,3, activation='relu', connection=layer4)
  return layer5

model = architectre(input_size)
print(' Input size = ' + str(input_size))
print('Output size = ' + str(model.size()))
print('---')
model.forward(np.random.rand(input_size['nc'], input_size['n2'], input_size['n1']))
print('---')
model.adjoint()
print('---')
  
