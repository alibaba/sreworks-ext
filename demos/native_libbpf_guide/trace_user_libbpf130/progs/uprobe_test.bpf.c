// SPDX-License-Identifier: GPL-2.0
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "common.h"
#include "uprobe_test.h"

SEC(".maps")
struct {
    __uint(type,        BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size,    sizeof(__u32));
    __uint(value_size,  sizeof(__u32));
    __uint(max_entries, MAX_CPUS);
} PERF_MAP;

SEC("uprobe")
int BPF_KPROBE(user_probe1, unsigned long long int x)
{
    struct event event = {};
    struct task_struct *task;
    struct task_struct *real_parent_task;

    const char uprobe_type[] = "func_uprobe1";
    memcpy(&event.cookie, uprobe_type, sizeof(event.cookie));

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

    event.var1      = x;
    event.var2      = x+5;

    int perf_ret = bpf_perf_event_output(ctx, &PERF_MAP, BPF_F_CURRENT_CPU, &event, sizeof(event));

   return 0;
}


SEC("uprobe//usr/bin/umark:func_uprobe2")
long user_probe2(struct pt_regs *ctx);

inline typeof(user_probe2(0)) ____user_probe2(struct pt_regs *ctx, unsigned long long int x, unsigned long long int y);  // can annotate 

inline typeof(user_probe2(0)) ____user_probe2(struct pt_regs *ctx, unsigned long long int x, unsigned long long int y)
{
    struct event event = {};
    struct task_struct *task;
    struct task_struct *real_parent_task;

    const char uprobe_type[] = "func_uprobe2";
    memcpy(&event.cookie, uprobe_type, sizeof(event.cookie));
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

    event.var1      = x;
    event.var2      = y;

    int perf_ret = bpf_perf_event_output(ctx, &PERF_MAP, BPF_F_CURRENT_CPU, &event, sizeof(event));

    return 0;
}

typeof(user_probe2(0)) user_probe2(struct pt_regs *ctx) {
    return ____user_probe2(ctx, (unsigned long long int)PT_REGS_PARM1(ctx), (unsigned long long int)PT_REGS_PARM2(ctx));
}

SEC("license") char _license[] = "GPL";
