// SPDX-License-Identifier: GPL-2.0
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "usdt.bpf.h"
#include "common.h"
#include "usdt_test.h"

SEC(".maps")
struct {
    __uint(type,        BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size,    sizeof(__u32));
    __uint(value_size,  sizeof(__u32));
    __uint(max_entries, MAX_CPUS);
} PERF_MAP;


SEC("usdt")
int BPF_USDT(usdt_probe1, int x){
    struct event event = {};
    struct task_struct *task;
    struct task_struct *real_parent_task;

    const char usdt_type[] = "func_usdt1";
    memcpy(&event.cookie, usdt_type, sizeof(event.cookie));

    event.micro_second = bpf_ktime_get_ns();

    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id;

    event.tgid = id >> 32;
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    event.uid = bpf_get_current_uid_gid();

    task = (struct task_struct *)bpf_get_current_task();
    bpf_probe_read_kernel(&real_parent_task, sizeof(real_parent_task), &task->real_parent);
    bpf_probe_read_kernel(&event.ppid,       sizeof(event.ppid),       &real_parent_task->pid);
    bpf_probe_read_kernel(&event.pcomm,      sizeof(event.pcomm),      &real_parent_task->comm);

    event.var1 = x;
    event.var2 = 17;

    int perf_ret = bpf_perf_event_output(ctx, &PERF_MAP, BPF_F_CURRENT_CPU, &event, sizeof(event));

    return 0;
}



SEC("usdt//usr/bin/umark:groupb:probe2")
int usdt_probe2(struct pt_regs *ctx);
static inline __attribute__((always_inline)) typeof(usdt_probe2(0)) ____usdt_probe2(struct pt_regs *ctx, int x, int y);

typeof(usdt_probe2(0)) usdt_probe2(struct pt_regs *ctx) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-conversion"
    return ____usdt_probe2(ctx, ({ long _x; bpf_usdt_arg(ctx, 0, &_x); (void *)_x; }), ({ long _x; bpf_usdt_arg(ctx, 1, &_x); (void *)_x; }));
#pragma GCC diagnostic pop
}
static inline __attribute__((always_inline)) typeof(usdt_probe2(0)) ____usdt_probe2(struct pt_regs *ctx, int x, int y)
{
    struct event event = {};
    struct task_struct *task;
    struct task_struct *real_parent_task;

    const char usdt_type[] = "func_usdt2";
    memcpy(&event.cookie, usdt_type, sizeof(event.cookie));
    event.micro_second = bpf_ktime_get_ns();

    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id;

    event.tgid = id >> 32;
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    event.uid = bpf_get_current_uid_gid();

    task = (struct task_struct *)bpf_get_current_task();
    bpf_probe_read_kernel(&real_parent_task, sizeof(real_parent_task), &task->real_parent);
    bpf_probe_read_kernel(&event.ppid,       sizeof(event.ppid),       &real_parent_task->pid);
    bpf_probe_read_kernel(&event.pcomm,      sizeof(event.pcomm),      &real_parent_task->comm);

    event.var1 = x;
    event.var2 = y;

    int perf_ret = bpf_perf_event_output(ctx, &PERF_MAP, BPF_F_CURRENT_CPU, &event, sizeof(event));

    return 0;
}

SEC("license") char _license[] = "GPL";
