#!/usr/bin/python
import sys
pre_load1 = float(sys.argv[1])
load1 = float(sys.argv[2])
load5s=(2048*(2048*load1-10)-1024-1884*2048*pre_load1)/(2048-1884)/2048
print(int(round(load5s)))