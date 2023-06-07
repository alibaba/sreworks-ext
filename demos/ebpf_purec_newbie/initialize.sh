#!/bin/bash

if [ $# -eq 0 ];then
    echo "kernel source dir is not set."       # like /tmp/kernel-4.18
    exit 22
elif [ $# -eq 1 ]; then
    src=$1
    target="/tmp/ebpf_project"                 # default /tmp/ebpf_project
else
    src=$1
    target=$2
fi

if [ ! -d "${src}" ];then
    echo "kernel source dir ${src} is incorrect."
    exit 2
fi
if [ -e "${target}" ];then
    echo "target dir ${target} is exists."
    exit 17
fi

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

mkdir ${target}
cd ${src}/tools/
rsync -amv --include '*/' --include '*.h' --exclude '*' arch include lib perf ${target}/tools
mkdir -p ${target}/helpers ${target}/tools/build/feature ${target}/tools/scripts
cp ${src}/tools/lib/bpf/*.c ${target}/tools/lib/bpf/
cd ${src}/tools/scripts/
cp Makefile.arch ${target}/tools/scripts/
cp Makefile.include ${target}/tools/scripts/
cd ${src}/scripts/
[ -e bpf_helpers_doc.py ] && cp bpf_helpers_doc.py ${target}/tools/scripts/
[ -e bpf_doc.py ] && cp bpf_doc.py ${target}/tools/scripts/
cd ${src}/samples/bpf/
cp trace_output_user.c trace_output_kern.c ${target}
[ -e bpf_load.c ] && cp bpf_load.c ${target}/helpers/
[ -e bpf_load.h ] && cp bpf_load.h ${target}/helpers/
[ -e asm_goto_workaround.h ] && cp asm_goto_workaround.h ${target}/helpers/
[ -e trace_common.h ] && cp trace_common.h ${target}/helpers/
cd ${src}/tools/build/feature
cp Makefile test-bpf.c test-libelf.c ${target}/tools/build/feature/
[ -e test-reallocarray.c ] && cp test-reallocarray.c ${target}/tools/build/feature/
cd ${src}/tools/testing/selftests/bpf
cp trace_helpers.c trace_helpers.h ${target}/helpers/
[ -e bpf_helpers.h ] && cp bpf_helpers.h ${target}/helpers/
cp ${SCRIPTPATH}/Makefile ${target}
[ -e ${SCRIPTPATH}/Makefile.libbpf ] && cp ${SCRIPTPATH}/Makefile.libbpf ${target}/tools/lib/bpf/Makefile || cp ${SCRIPTPATH}/tools/lib/bpf/Makefile ${target}/tools/lib/bpf/
cp ${SCRIPTPATH}/initialize.sh ${target}
