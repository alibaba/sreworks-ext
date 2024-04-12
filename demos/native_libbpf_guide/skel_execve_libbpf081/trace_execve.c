#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>
#include <linux/limits.h>
#include <linux/perf_event.h>
#include <sys/resource.h>
#include <linux/ring_buffer.h>
#include <bpf/libbpf.h>

#include "common.h"
#include "trace_execve.h"
#include "skeleton.skel.h"

#define MAP_PAGE_SIZE 1024
#define MAX_CNT 1000000llu

static __u64  cnt;
static int    event_map_fd = 0;
static struct bpf_object  *bpf_obj  = NULL;
static struct bpf_program *bpf_prog = NULL;
static struct bpf_link    *bpf_link = NULL;

static void print_bpf_output(void *ctx, int cpu, void *data, __u32 size)
{
    struct event* v = (struct event *)data;
    printf("%s %llu %u %s %u %s %u %s\n", v->cookie, (unsigned long long)v->micro_second, (unsigned int)v->tgid, v->comm, (unsigned int)v->ppid, v->pcomm, v->uid, v->fname);

    cnt++;
    if (cnt == MAX_CNT) {
        printf("recv %llu events\n",   MAX_CNT);
    }
}

void handle_lost_events(void *ctx, int cpu, __u64 lost_cnt)
{
    printf("Lost %llu events on CPU #%d!\n", lost_cnt, cpu);
}

int main(int argc, char *argv[])
{
    struct rlimit lim = {
        .rlim_cur = RLIM_INFINITY,
        .rlim_max = RLIM_INFINITY,
    };

    setrlimit(RLIMIT_MEMLOCK, &lim);

    bpf_obj = bpf_object__open_mem(obj_buf, obj_buf_sz, NULL);
    if (libbpf_get_error(bpf_obj)) {
        printf("ERROR: failed to open prog: '%s'\n", strerror(errno));
        return 1;
    }

    if (bpf_object__load(bpf_obj)) {
        printf("ERROR: failed to load prog: '%s'\n", strerror(errno));
        return 1;
    }

    bpf_prog = bpf_object__find_program_by_name(bpf_obj,"trace_execve_enter");
    bpf_link = bpf_program__attach_tracepoint(bpf_prog, "syscalls", "sys_enter_execve");
    if (libbpf_get_error(bpf_link)) {
        return 2;
    }

    event_map_fd = bpf_object__find_map_fd_by_name(bpf_obj, "perf_map");
    if ( 0 >= event_map_fd){
        printf("ERROR: failed to load event_map_fd: '%s'\n", strerror(errno));
        return 1;
    }

    struct perf_buffer *pb;
    int ret;

    pb = perf_buffer__new(event_map_fd, MAP_PAGE_SIZE, print_bpf_output, handle_lost_events, NULL, NULL);
    ret = libbpf_get_error(pb);
    if (ret) {
        printf("ERROR: failed to setup perf_buffer: %d\n", ret);
        return 1;
    }

    while ((ret = perf_buffer__poll(pb, 1000)) >= 0 ) {
        // go forever
    }

    bpf_link__destroy(bpf_link);
    bpf_object__close(bpf_obj);

    return 0;
}

