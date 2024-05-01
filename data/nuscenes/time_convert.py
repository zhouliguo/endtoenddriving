import time
import datetime


timestamp = time.time()
print(timestamp)
#dt = datetime.datetime.fromtimestamp(timestamp)
#date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
#print(date_str)

dt = datetime.datetime.fromtimestamp(timestamp)

date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
date_str = dt.strftime('%Y-%m-%d-%H-%M-%S')
print(date_str)