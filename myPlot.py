import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
matplotlib.use('Agg')
file = open(sys.argv[1])
data = file.readlines()
title = data[0]
xlabel = "epoch"
ylabel = data[1]
train_d = [float(i) for i in data[2].split(",")]
test_d = [float(i) for i in data[3].split(",")]


plt.plot(train_d,label="train")
plt.plot(test_d, label="test")
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.legend()
plt.savefig("plot"+title[:-5]+".png")
