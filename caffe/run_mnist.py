import os

def manist():
    cmd_list = [
       "examples/mnist/train_lenet.sh",
       "examples/mnist/train_lenet_consolidated.sh",
       "examples/mnist/train_mnist_autoencoder_adagrad.sh",
       "examples/mnist/train_mnist_autoencoder_nesterov.sh",
       "examples/mnist/train_mnist_autoencoder.sh",
    ]

    def get_name(path):
        return path.split("/")[-1].split(".")[0]

    def exec_cmd(cmd):
        print cmd
        os.system("sh ./{} 2>&1 |tee {}".format(cmd,get_name(cmd)))
    
    print "begin run\n"
    map(exec_cmd,cmd_list)


def main():
    
    manist()
      

if __name__ == "__main__":
    main()
