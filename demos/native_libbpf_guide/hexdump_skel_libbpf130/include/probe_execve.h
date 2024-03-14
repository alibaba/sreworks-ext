/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
#ifndef __EXECVE_PROBE_H
#define __EXECVE_PROBE_H

struct event {
    pid_t tgid;
    pid_t ppid;
    uid_t uid;
    __u64 micro_second;
    char cookie[16];
    char comm[TASK_COMM_LEN];
    char pcomm[TASK_COMM_LEN];
    char fname[NAME_MAX];
};

#endif /* __EXECVE_PROBE_H */
