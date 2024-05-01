import datetime
import time
 
timestamp = time.time() #1636497673
print(timestamp)
dt = datetime.datetime.fromtimestamp(timestamp)
date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
print(date_str)