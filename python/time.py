import time
from datetime import datetime
'''
time.time()    
    Get the current timestamp.
datetime.datetime
    Get or translate time into readable time string.
    now()
    fromtimesstamp(timestamp)
'''
print(time.time())
print(datetime.now())
print(datetime.fromtimestamp(time.time()))