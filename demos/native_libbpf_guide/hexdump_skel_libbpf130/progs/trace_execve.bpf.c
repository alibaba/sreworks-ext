// SPDX-License-Identifier: GPL-2.0
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "common.h"
#include "trace_execve.h"

struct syscalls_enter_execve_args {
    struct trace_entry ent;
    int                __syscall_nr;
    const char *       filename;
    const char *const *argv;
    const char *const *envp;
};

SEC(".maps")
struct {
    __uint(type,        BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size,    sizeof(__u32));
    __uint(value_size,  sizeof(__u32));
    __uint(max_entries, MAX_CPUS);
} PERF_MAP;

SEC("tracepoint")
int trace_execve_enter(struct syscalls_enter_execve_args *ctx){
    struct event event = {};
    struct task_struct *task;
    struct task_struct *real_parent_task;

    const char execve_type[] = "trace_execve";
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

    bpf_probe_read_kernel(&event.fname, sizeof(event.fname), (void *)ctx->filename);

    int perf_ret = bpf_perf_event_output(ctx, &PERF_MAP, BPF_F_CURRENT_CPU, &event, sizeof(event));

    return 0;
}

SEC("license") char _license[] = "GPL";
