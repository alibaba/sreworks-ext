/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
#ifndef __UPROBE_H
#define __UPROBE_H

struct event {
    int var1;
    int var2;
    pid_t tgid;
    pid_t ppid;
    uid_t uid;
    __u64 micro_second;
    char cookie[16];
    char comm[TASK_COMM_LEN];
    char pcomm[TASK_COMM_LEN];
};

#endif /* __UPROBE_H */
