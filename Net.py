import numpy as np

class Net():
	def __init__(self, trainingInputs, trainingOutputs, testInputs, testOutputs):
		self.trainingInputs = trainingInputs
		self.trainingOutputs = trainingOutputs
		self.testInputs = testInputs
		self.testOutputs = testOutputs

		self.maxAcc = 0.0

		self.layer1 = ReLU(784,300)
		self.layer2 = dropOut(300,0.30)
		self.layer3 = Sigmoid(300,10)

	def feedforward(self, data, backprop):
		self.layer1.feedforward(data)
		self.layer2.feedforward(self.layer1.act, backprop)
		self.layer3.feedforward(self.layer2.act)

	def backprop(self, batchSize, LR,count):
		time = 0

		suml3wGrad = 0
		suml3bGrad = 0
		suml1wGrad = 0
		suml1bGrad = 0
		for time in range(len(self.trainingInputs)):
			x = np.random.randint(0,len(self.trainingInputs))
			inputs = np.matrix(self.trainingInputs[x][np.newaxis])
			outputs = np.matrix(self.trainingOutputs[x])
			self.feedforward(inputs, True)
			cost = self.layer3.act - outputs
			l3wupdate, l3bupdate = self.layer3.backprop(cost, self.layer2.act, False)
			self.layer2.backprop(cost, self.layer1.act, self.layer3.delta)
			l1wupdate, l1bupdate = self.layer1.backprop(cost, inputs, self.layer2.delta)
			suml3wGrad += l3wupdate
			suml3bGrad += l3bupdate
			suml1wGrad += l1wupdate
			suml1bGrad += l1bupdate
			time += 1
			if time % batchSize == 0:
				self.layer3.weights = self.layer3.weights - self.SGD(suml3wGrad, batchSize, LR)
				self.layer3.biases = self.layer3.biases - self.SGD(suml3bGrad,batchSize,LR)
				self.layer1.weights = self.layer1.weights - self.SGD(suml1wGrad,batchSize,LR)
				self.layer1.biases = self.layer1.biases - self.SGD(suml1bGrad,batchSize,LR)
				suml3wGrad = 0
				suml3bGrad = 0
				suml1wGrad = 0
				suml1bGrad = 0
			if time % 50 == 0:
				print(time/50.0 + (count*800))
				print("Accuracy = ", self.acc(zip(self.testInputs, self.testOutputs),True))
				cost = self.cost(self.layer3.act, self.testOutputs)
				print("Cost = ", cost)

	#Optimizers

	def SGD(self, gradUpdate, batchSize, LR):
		return gradUpdate/batchSize*LR


	#Indicators

	def cost(self, a, y):
		return np.sum(a - y) ** 2

	def acc(self, data, save):
		result = 0
		total = 0
		for x,y in data:
			self.feedforward(x, False)
			if np.argmax(self.layer3.act) == np.argmax(y):
				result += 1.0
			total += 1.0
		return result/total

#Layers

class Sigmoid():
	def __init__(self, prevLaySize, size):
		self.weights = np.random.rand(prevLaySize,size)/(np.sqrt(1000/(prevLaySize)))
		self.biases = np.zeros((1,size))

	def feedforward(self, act):
		self.z = act.dot(self.weights) + self.biases
		self.act = 1.0 / (1.0 + np.exp(np.clip(-self.z, -500,500)))

	def backprop(self, cost, prevAct, forwardGrad = False):
		prime = (1.0 - (1.0 / (1.0 + np.exp(-self.z)))) * 1.0 / (1.0 + np.exp(-self.z))
		if type(forwardGrad) == type(False):
			self.delta = np.multiply(prime, cost)
		else:
			self.delta = np.multiply(prime, forwardGrad.transpose())
		wGrad = np.dot(prevAct.transpose(), self.delta)
		bGrad = self.delta
		self.delta = np.dot(self.weights, self.delta.transpose())
		return wGrad, bGrad

class TanH():
	def __init__(self, prevLaySize, size):
		self.weights = np.random.rand(prevLaySize,size)/(np.sqrt(1/prevLaySize))
		self.biases = np.zeros((1,size))

	def feedforward(self, act):
		self.z = act.dot(self.weights) + self.biases
		self.act = np.tanh(self.z)

	def backprop(self, cost, prevAct, forwardGrad = False):
		prime = 1.0 - np.square(np.tanh(self.z))
		if type(forwardGrad) == type(False):
			self.delta = np.multiply(prime, cost)
		else:
			self.delta = np.multiply(prime, forwardGrad.transpose())
		wGrad = np.dot(prevAct.transpose(), self.delta)
		bGrad = self.delta
		self.delta = np.dot(self.weights, self.delta.transpose())
		return wGrad, bGrad

class ReLU():
	def __init__(self, prevLaySize, size):
		self.weights = (0.0001*np.random.rand(prevLaySize,size))/(np.sqrt(2/prevLaySize))
		self.biases = np.zeros((1,size))

	def feedforward(self, act):
		self.z = act.dot(self.weights) + self.biases
		self.act = np.maximum(self.z,0)

	def backprop(self, cost, prevAct, forwardGrad = False):
		prime = np.zeros((self.act.shape))
		prime[self.z >= 0] = 1.0
		if type(forwardGrad) == type(False):
			self.delta = np.multiply(prime, cost)
		else:
			self.delta = np.multiply(prime, forwardGrad.transpose())
		wGrad = np.dot(prevAct.transpose(), self.delta)
		bGrad = self.delta
		self.delta = np.dot(self.weights, self.delta.transpose())
		return wGrad, bGrad

class dropOut():
	def __init__(self, size, percDropout):
		self.percDropout = percDropout
		self.size = size

	def feedforward(self, act, backprop):
		if backprop:
			self.dropOutMat = (self.percDropout < np.random.rand(1,self.size))
			self.act = np.multiply(self.dropOutMat, act)
		else:
			self.act = act

	def backprop(self, cost, prevAct, forwardGrad = False):
		self.delta = np.multiply(forwardGrad, self.dropOutMat.transpose())
	

threshold = 40000

inputTrain = inputData[:threshold]
outputTrain = outputData[:threshold]
inputTest = inputData[threshold:]
outputTest = outputData[:threshold]

net = Net(inputTrain, outputTrain, inputTest, outputTest)
for x in range(1000000):
	net.backprop(50,1,x)