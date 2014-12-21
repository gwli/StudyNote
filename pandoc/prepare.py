import subprocess

libs = [
 'haskell-platform',
 'texlive',
]

def install_libs(package):
    print("##BEGIN install:{} \n".format(package))
    subprocess.check_call(['apt-get', '-y','install',package])
    print("##End install:{} \n".format(package))

cabal_libs= [
    'pandoc',
    'pandoc-citeproc',
    'hsb2hs',
]

def install_pandoc(package):
    print("##BEGIN install:{} \n".format(package))
    subprocess.check_call(['cabal','update'])
    subprocess.check_call(['cabal','install',package])
    print("##End install:{} \n".format(package))


def main():
    map(install_libs,libs)
    map(install_pandoc,cabal_libs)

if __name__ == "__main__":
    main()
