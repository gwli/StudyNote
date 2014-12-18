import os

root= "/home/devtoolsqa8/myown/trunk/DL/svm/libsvm"

"2^5, and 2^-7"
cmd = "{}/svm-train -c 32 -g 0.0078125 mnist.scale mnist.scale.module".format(root)
print(cmd)

os.system(cmd)
