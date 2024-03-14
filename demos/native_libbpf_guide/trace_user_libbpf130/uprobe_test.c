#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>
#include <linux/limits.h>
#include <linux/perf_event.h>
#include <sys/resource.h>
#include <linux/ring_buffer.h>
#include <bpf/libbpf.h>

#include "uprobe_helper.h"
#include "common.h"
#include "uprobe_test.h"

#include "skeleton.skel.h"

#define MAP_PAGE_SIZE 1024
#define MAX_CNT 1000000llu

static __u64  cnt;
static int    event_map_fd = 0;
static struct bpf_object  *bpf_obj  = NULL;
static struct bpf_program *bpf_prog1 = NULL;
static struct bpf_link    *bpf_link1 = NULL;
static struct bpf_program *bpf_prog2 = NULL;
static struct bpf_link    *bpf_link2 = NULL;

static void print_bpf_output(void *ctx, int cpu, void *data, __u32 size)
{
    struct event* v = (struct event *)data;
    printf("%s %llu %u %s %u %s %u %llu %llu\n", v->cookie, (unsigned long long)v->micro_second, (unsigned int)v->tgid, v->comm, (unsigned int)v->ppid, v->pcomm, v->uid,  (unsigned long long)v->var1, (unsigned long long)v->var2);

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
    off_t func_off1;

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

    func_off1 = get_elf_func_offset("/usr/bin/umark", "func_uprobe1");
    bpf_prog1 = bpf_object__find_program_by_name(bpf_obj, "user_probe1");
    bpf_link1 = bpf_program__attach_uprobe(bpf_prog1, 0, -1, "/usr/bin/umark", func_off1);
    if (libbpf_get_error(bpf_link1)) {
	printf("ERROR: failed to attach_uprobe1: '%s'\n", strerror(errno));
        return 2;
    }

    bpf_prog2 = bpf_object__find_program_by_name(bpf_obj, "user_probe2");
    bpf_link2 = bpf_program__attach(bpf_prog2);
    if (libbpf_get_error(bpf_link2)) {
	printf("ERROR: failed to attach_uprobe2: '%s'\n", strerror(errno));
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

    bpf_link__destroy(bpf_link1);
    bpf_link__destroy(bpf_link2);
    bpf_object__close(bpf_obj);

    return 0;
}
