import os

cmd_sys = [
  "uname -a" ,#check kernel and os
  "head -n 1 /etc/issue",
  "hostname",
  "lspci -tv",
  "lsusb -tv",
  "lsmod",
  "env",
]



def run_cmd(cmd):
    print("###${}".format(cmd))
    os.system(cmd)

def check_system():
    map(run_cmd,cmd_sys)


cmd_resource = [
    "free -m",
    "df -h",
    "du -sh",
    "grep MemTotal /proc/meminfo",
    "grep MemFree /proc/meminfo",
    "uptime",
    "cat /proc/loadavg",
]


def check_ressource():
    map(run_cmd,cmd_resource)


cmd_partiton = [
    "mount |column -t",
    "fdisk -l",
    "swapon -a",
    "hdparm -i /dev/sda",
    "lsof",
]

def check_partition():
    map(run_cmd,cmd_partiton)



cmd_network = [
    "ifconfig",
    "iptable -L",
    "route -n"
    "netstat -lntp",
    "netstat -antp",
    "netstat -s"
]

def check_network():
    map(run_cmd,cmd_network)

cmd_ps = [
    "ps -ef",
    #"top",
    "lsof",
]


def check_ps():
    map(run_cmd,cmd_ps)


cmd_users = [
   "w",
   "id",
   "last",
   "cut -d: -f1 /etc/passwd",
   "cut -d: -f1 /etc/group",
]    

def check_users():
    map(run_cmd,cmd_users)

cmd_services = [
    "chkconfig --list",
    "chkconfig --list |grep on",
    "crontab -l",
]

def check_service():
    map(run_cmd,cmd_services)


def batch():
    check_system()
    check_ressource()
    check_partition()
    check_network()
    check_ps()
    check_users()
    check_service()

if __name__ == "__main__":
   batch()

