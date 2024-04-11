// SPDX-License-Identifier: GPL-2.0
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "common.h"
#include "probe_execve.h"

#if defined(__TARGET_ARCH_x86)
#define SYS_STAT_NAME "__x64_sys_execve"
#elif defined(__TARGET_ARCH_arm64)
#define SYS_STAT_NAME "__arm64_sys_execve"
#else
#error "Unsupported architecture"
#endif

SEC(".maps")
struct {
    __uint(type,        BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size,    sizeof(__u32));
    __uint(value_size,  sizeof(__u32));
    __uint(max_entries, MAX_CPUS);
} PERF_MAP;

SEC("kprobe/"SYS_STAT_NAME)
int sys_execve_enter(struct pt_regs *ctx){
    struct event event = {};
    struct task_struct *task;
    struct task_struct *real_parent_task;

    const char execve_type[] = "probe_execve";
    memcpy(&event.cookie, execve_type, sizeof(event.cookie));
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

    int perf_ret = bpf_perf_event_output(ctx, &PERF_MAP, BPF_F_CURRENT_CPU, &event, sizeof(event));

    return 0;
}

SEC("license") char _license[] = "GPL";
