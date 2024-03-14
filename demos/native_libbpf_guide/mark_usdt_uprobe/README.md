sudo bpftrace -e 'usdt:/usr/bin/umark:groupb:probe2 { printf("arg value: %d %d\n", arg0, arg1); }'  -p $(pidof umark)
