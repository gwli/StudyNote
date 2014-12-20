import subprocess as sp
libs = [
    "libatlas-base-dev",
    "python-dev",
    "libprotobuf-dev", "libleveldb-dev", "libsnappy-dev", "libopencv-dev", "libboost-all-dev", "libhdf5-serial-dev",
    "libgflags-dev", "libgoogle-glog-dev", "liblmdb-dev", "protobuf-compiler",
]


def install(package):
    print("### Begin Install package:{}".format(package))
    sp.check_call(['apt-get','-y','install',package]) 
    print("### End Install package:{}".format(package))


map(install,libs)
  
