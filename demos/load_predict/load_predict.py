#!/usr/bin/python
import sys
last_load = float(sys.argv[1])
active=int(sys.argv[2])
exp=1884
last_load_l = int(2048 * last_load)
active_long = 2048 * active
load = last_load_l * exp
load = load + active_long * (2048 - exp)
load = load + 1024
load = load / 2048
load = load + 10
print("%.1f"%(float(load) / 2048))
