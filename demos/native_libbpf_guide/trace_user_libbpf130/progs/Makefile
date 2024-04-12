# SPDX-License-Identifier: GPL-2.0

LLC          ?= llc
CLANG        ?= clang
OPT          ?= opt
LLVM_DIS     ?= llvm-dis
LLVM_OBJCOPY ?= llvm-objcopy
BTF_PAHOLE   ?= pahole
BPFTOOL      ?= bpftool
ROOTDIR      ?= ./../

BPF_OBJECT   := $(if $(BPF_TARGET),$(BPF_TARGET),$(patsubst %.c,%.o,$(wildcard *.bpf.c)))
VERIFY       := $(if $(BPF_TARGET),,verify_target_bpf)

include $(ROOTDIR)/tools/scripts/Makefile.arch

KERNELV      :=$(shell uname -r)
KERNEL-DEVEL := /lib/modules/$(KERNELV)/build/

MACHINE_NAME := $(shell uname -m)
MACHINE_DIR  := /usr/include/$(MACHINE_NAME)-linux-gnu

DIRAFTER_FLAGS += -idirafter $(shell $(CLANG) -print-file-name=include)
DIRAFTER_FLAGS += -idirafter /usr/include
ifeq ($(shell test -d $(MACHINE_DIR) && echo -n yes),yes)
    DIRAFTER_FLAGS += -idirafter $(MACHINE_DIR)
endif

VMLINUX_HEADER := $(ROOTDIR)/tools/lib/vmlinux.h

BTF_LLC_PROBE     := $(shell $(LLC) -march=bpf -mattr=help 2>&1 | grep dwarfris)
BTF_PAHOLE_PROBE  := $(shell $(BTF_PAHOLE) --help 2>&1 | grep BTF)
BTF_OBJCOPY_PROBE := $(shell $(LLVM_OBJCOPY) --help 2>&1 | grep -i 'usage.*llvm')
BTF_LLVM_PROBE    := $(shell echo "int main() { return 0; }"|$(CLANG) -target bpf -O2 -g -c -x c - -o ./llvm_btf_verify.o;readelf -S ./llvm_btf_verify.o | grep BTF)

BPF_EXTRA_CFLAGS += -g -fno-stack-protector

try-run = $(shell set -e;if ($(1)) >/dev/null 2>&1;then echo "$(2)";else echo "$(3)";fi)
cc-option-yn = $(call try-run, $(CC) -Werror $(KBUILD_CFLAGS) $(1) -x c -c /dev/null -o /dev/null,y,n)

NOWARN += -Wno-compare-distinct-pointer-types

NOFLAG += -fno-asynchronous-unwind-tables
NOFLAG += -fno-stack-protector

CLANG_VERSION=$(shell echo | clang -dM -E - | grep __clang_major__ | awk '{print $$3}')

VMLINUX_NORMAL:=/sys/kernel/btf/vmlinux
VMLINUX_CLASSIC:=/boot/vmlinux-$(KERNELV)

ifneq ($(wildcard $(VMLINUX_NORMAL)),)
  VMLINUX_BTF:=$(VMLINUX_NORMAL)
else
  ifneq ($(wildcard $(VMLINUX_CLASSIC)),)
    VMLINUX_BTF:=$(VMLINUX_CLASSIC)
  endif
endif

.PHONY: clean $(CLANG) $(LLC) $(BPFTOOL)

clean:
	rm -f *.ll *.o *.d .*.d $(VMLINUX_HEADER)

verify_cmds: $(CLANG) $(LLC) $(BPFTOOL)
	for TOOL in $^ ; do \
                if ! (which -- "$${TOOL}" > /dev/null 2>&1); then \
                        echo "*** ERROR: Cannot find LLVM tool $${TOOL}" ;\
                        exit 1; \
                else true; fi; \
        done

verify_target_bpf: verify_cmds
	if ! (${LLC} -march=bpf -mattr=help > /dev/null 2>&1); then \
                echo "*** ERROR: LLVM (${LLC}) does not support 'bpf' target" ;\
                echo "   NOTICE: LLVM version >= 3.7.1 required" ;\
                exit 2; \
        else true; fi

verify_btf:
	if [ -z "$(VMLINUX_BTF)" ]; then \
                echo "*** ERROR: Both $(VMLINUX_NORMAL) and $(VMLINUX_CLASSIC) do not exist" ;\
                echo "   NOTICE: BTF file is required" ;\
                exit 2; \
        else true; fi
	
$(VMLINUX_HEADER): verify_btf
	$(BPFTOOL) btf dump file $(VMLINUX_BTF) format c > $(VMLINUX_HEADER)

$(BPF_OBJECT): %.o: $(VMLINUX_HEADER) %.c
	$(CLANG) -iquote $(ROOTDIR)/include/ -iquote $(ROOTDIR)/helpers -I$(ROOTDIR)/tools/lib/ -I$(ROOTDIR)/tools/include/uapi $(DIRAFTER_FLAGS) -DENABLE_ATOMICS_TESTS -D__KERNEL__ -D__BPF_TRACING__ -D__TARGET_ARCH_$(ARCH) $(CC_FLAGS_USING) $(BPF_EXTRA_CFLAGS) -Werror $(NOWARN) $(NOFLAG)  -O2 -mlittle-endian -target bpf -mcpu=v3 -c $(word 2,$^) -o $@

all: $(VERIFY) $(BPF_OBJECT)

.DEFAULT_GOAL := all
