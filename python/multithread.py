import threading
import time

class myThread(threading.Thread):
    def __init__(self,name):
        threading.Thread.__init__(self)
        self.name=name
        self.names=[]
    def run(self):
        print('{} run'.format(self.name))
        time.sleep(2)
        print('{} stop'.format(self.name))
        self.names.append(['hello'+self.name])

thread1=myThread('1')
thread2=myThread('2')
thread1.start()
thread1.join()
thread2.start()
thread2.join()
names=[]
names.extend(thread1.names)
names.extend(thread2.names)
print(names)