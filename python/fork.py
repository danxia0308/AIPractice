import os
print('Parent pid {}'.format(os.getpid()))
for i in range(3):
    pid=os.fork()
    if pid==0:
        print('children, pid={}',os.getpid())
    else:
        print('parent, pid={}, children={}'.format(os.getpid(), pid))
    print('pid={},result={}'.format(os.getpid(),i*i))