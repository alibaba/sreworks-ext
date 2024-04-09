#include <unistd.h>
#include <stdio.h>
//#include <sys/sdt.h>
#include "sdt.h"

unsigned long long int func_uprobe1(unsigned long long int x){
    return x + 1;
}
unsigned long long int func_uprobe2(unsigned long long int x, unsigned long long int y){
    return x + y;
}

int main(int argc, char const *argv[]) {
    unsigned long long int i;
    int var1 = 10;
    int var2 = 20;
    int var3 = 30;

    for (i = 0; i < 86400000; i++) {
        sleep(1);
       
        DTRACE_PROBE1(groupa, probe1, var1);
        DTRACE_PROBE2(groupb, probe2, var2, var3);
        printf("hit uprobe1 %llu\n", func_uprobe1(i));
        printf("hit uprobe2 %llu\n", func_uprobe2(i + 3, i + 8));
    }

    return 0;
}
