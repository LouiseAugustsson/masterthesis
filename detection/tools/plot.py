
#Script for plotting training loss VS number of iterations
#solve.log.test plots accuracy, solve.log.train plots loss
#Do not forget to change y-label! 

import matplotlib.pyplot as plt

iterationS = []
lossS = []
iterationP = []
lossP = [] 
countS = 0
countP = 0
max_plot = 100000

with open('solve.log.train', 'r') as  log:
	for line in log:
		countS = countS + 1
		if line[0] != '#':
			line = line.split()
			if countS <= max_plot:
				iterationS.append(float(line[0]))
				lossS.append(float(line[2]))

# with open('solve.log.test', 'r') as  log:
# 	for line in log:
# 		countP = countP + 1 
# 		if line[0] != '#':
# 			line = line.split()
# 			if countP <= max_plot:
# 				iterationP.append(float(line[0]))
# 				lossP.append(float(line[2]))


plt.plot( iterationS, lossS) #, iterationP, lossP)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Loss', fontsize = 18)
#plt.axis((-100,10000, 0, 7000))
#plt.legend(['Traing', 'Testing'], fontsize = 15)
plt.show()

