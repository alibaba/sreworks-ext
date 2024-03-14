/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
#ifndef __COMMON_H
#define __COMMON_H

#define TASK_COMM_LEN 16
#define NAME_MAX 255
#define MAX_CPUS 128

#define STR_VALUE(arg)   #arg
#define STRINGIFY(name)  STR_VALUE(name)

#define PERF_MAP      perf_map
#define PERF_MAP_NAME STRINGIFY(PERF_MAP)

#define memcpy(dest, src, n)   __builtin_memcpy((dest), (src), (n))

#ifndef BPF_F_CURRENT_CPU
#define BPF_F_CURRENT_CPU ((u32)-1)
#endif

#define bpf_printk2(fmt, ...)                                   \
({                                                              \
               char ____fmt[] = fmt;                            \
               bpf_trace_printk(____fmt, sizeof(____fmt),       \
                                ##__VA_ARGS__);                 \
})

typedef enum bpf_perf_event_ret( * perf_event_print_fn)(void * data, int size);

struct perf_event_sample {
    struct perf_event_header header;
    __u32 size;
    char data[];
};

#endif /* __COMMON_H */
