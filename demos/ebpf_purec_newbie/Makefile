# SPDX-License-Identifier: GPL-2.0

CC     = $(CROSS_COMPILE)gcc
LD     = $(CROSS_COMPILE)ld
AS     = $(CROSS_COMPILE)as

HELPERS_PATH := ./helpers
TOOLS_PATH   := ./tools

CFLAGS += -I./tools/lib/
CFLAGS += -I./tools/lib/bpf/
CFLAGS += -I./tools/include/
CFLAGS += -I./tools/perf/
#CFLAGS += -DHAVE_ATTR_TEST=0

comma   := ,
dot-target = $(dir $@).$(notdir $@)
depfile = $(subst $(comma),_,$(dot-target).d)

HELPERINCLUDE = -I$(HELPERS_PATH)

LIBBPF   = $(TOOLS_PATH)/lib/bpf/libbpf.a
LDLIBS  += $(LIBBPF) -lelf -lz -lrt

LOADER   = trace_output
USER_SRC = trace_output_user.c
BPF_SRC  = trace_output_kern.c

USER_OBJECT = $(patsubst %.c,%.o,$(USER_SRC))
BPF_OBJECT  = $(patsubst %.c,%.o,$(BPF_SRC))

HELPER_OBJECTS  = $(patsubst %.c,%.o,$(wildcard $(HELPERS_PATH)/*.c))

############################################################

LLC          ?= llc
CLANG        ?= clang
OPT          ?= opt
LLVM_DIS     ?= llvm-dis
LLVM_OBJCOPY ?= llvm-objcopy
BTF_PAHOLE   ?= pahole

include ./tools/scripts/Makefile.arch

NOSTDINC_FLAGS      += -nostdinc -isystem $(shell $(CC) -print-file-name=include)

KERNELV=$(shell uname -r)
KERNEL-DEVEL += /lib/modules/$(KERNELV)/build/

USERINCLUDE    += -I$(KERNEL-DEVEL)/arch/$(SRCARCH)/include/uapi
USERINCLUDE    += -I$(KERNEL-DEVEL)/arch/$(SRCARCH)/include/generated/uapi
USERINCLUDE    += -I$(KERNEL-DEVEL)/include/uapi
USERINCLUDE    += -I$(KERNEL-DEVEL)/include/generated/uapi
USERINCLUDE    += -include $(KERNEL-DEVEL)/include/linux/kconfig.h

LINUXINCLUDE   += -I$(KERNEL-DEVEL)/arch/$(SRCARCH)/include 
LINUXINCLUDE   += -I$(KERNEL-DEVEL)/arch/$(SRCARCH)/include/generated
LINUXINCLUDE   += -I$(KERNEL-DEVEL)/include
LINUXINCLUDE   += $(USERINCLUDE)

BTF_LLC_PROBE     := $(shell $(LLC) -march=bpf -mattr=help 2>&1 | grep dwarfris)
BTF_PAHOLE_PROBE  := $(shell $(BTF_PAHOLE) --help 2>&1 | grep BTF)
BTF_OBJCOPY_PROBE := $(shell $(LLVM_OBJCOPY) --help 2>&1 | grep -i 'usage.*llvm')
BTF_LLVM_PROBE    := $(shell echo "int main() { return 0; }"|$(CLANG) -target bpf -O2 -g -c -x c - -o ./llvm_btf_verify.o;readelf -S ./llvm_btf_verify.o | grep BTF)

BPF_EXTRA_CFLAGS += -fno-stack-protector
ifneq ($(BTF_LLVM_PROBE),)
        BPF_EXTRA_CFLAGS += -g
else
ifneq ($(and $(BTF_LLC_PROBE),$(BTF_PAHOLE_PROBE),$(BTF_OBJCOPY_PROBE)),)
        BPF_EXTRA_CFLAGS += -g
        LLC_FLAGS += -mattr=dwarfris
        DWARF2BTF = y
endif
endif

ifdef CROSS_COMPILE
CLANG_ARCH_ARGS = --target=$(notdir $(CROSS_COMPILE:%-=%))
endif

NOWARN += -Wno-unused-value 
NOWARN += -Wno-pointer-sign  
NOWARN += -Wno-compare-distinct-pointer-types 
NOWARN += -Wno-gnu-variable-sized-type-not-at-end 
NOWARN += -Wno-address-of-packed-member 
NOWARN += -Wno-tautological-compare 
NOWARN += -Wno-unknown-warning-option 

NOFLAG += -fno-asynchronous-unwind-tables

bpf_helpers_doc:=$(wildcard ./tools/scripts/bpf_helpers_doc.py)
ifeq ($(bpf_helpers_doc),)
  bpf_helpers_doc:=$(wildcard ./tools/scripts/bpf_doc.py)
endif
python3_file:=$(wildcard /usr/bin/python3)
ifneq ($(and $(bpf_helpers_doc),$(python3_file)),)
    bpf_helpers_doc_header:=$(shell $(bpf_helpers_doc) --help | grep -P "\-\-header")
    ifneq ($(bpf_helpers_doc_header),)
        BPF_HELPER_DEFS := ./tools/lib/bpf/bpf_helper_defs.h
    endif
endif

CLANG_VERSION=$(shell echo | clang -dM -E - | grep __clang_major__ | awk '{print $$3}')

ifeq ($(shell expr $(CLANG_VERSION) '<' 11), 1)
  ASM_GOTO_FLAG = -include asm_goto_workaround.h
endif

.PHONY: clean bpf_prog $(CLANG) $(LLC)

clean:
	rm -f *.ll *.o *.d .*.d $(LOADER) $(HELPERS_PATH)/*.d $(HELPERS_PATH)/*.o $(HELPERS_PATH)/.*.d llvm_btf_verify.o $(BPF_HELPER_DEFS)
	make -C ./tools/lib/bpf/ clean
	make -C ./tools/build/feature clean

$(BPF_HELPER_DEFS):
	$(bpf_helpers_doc) --header --file ./tools/include/uapi/linux/bpf.h > $(BPF_HELPER_DEFS)

verify_cmds: $(CLANG) $(LLC)
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

$(BPF_OBJECT): %.o: %.c
	$(CLANG) $(NOSTDINC_FLAGS) $(LINUXINCLUDE) $(BPF_EXTRA_CFLAGS) -I./tools/lib/ -D__KERNEL__ -D__BPF_TRACING__ -D__TARGET_ARCH_$(ARCH) $(NOWARN) $(NOFLAG) $(CLANG_ARCH_ARGS) $(HELPERINCLUDE) -I./tools/lib/bpf/ $(ASM_GOTO_FLAG) -O2 -emit-llvm -Xclang -disable-llvm-passes -c $< -o - | \
        $(OPT) -O2 -mtriple=bpf-pc-linux | \
        $(LLVM_DIS) | \
        $(LLC) -march=bpf $(LLC_FLAGS) -filetype=obj -o $@
ifeq ($(DWARF2BTF),y)
	$(BTF_PAHOLE) -J $@
endif

bpf_prog: verify_target_bpf $(BPF_HELPER_DEFS) $(BPF_OBJECT)

$(LIBBPF):
	make -C ./tools/lib/bpf/

$(USER_OBJECT): %.o:%.c
	$(CC) -Wp,-MD,$(depfile) $(CFLAGS) $(HELPERINCLUDE) -c -o $@ $<

$(HELPER_OBJECTS): %.o:%.c
	$(CC) -Wp,-MD,$(depfile) $(CFLAGS) $(HELPERINCLUDE) -c -o $@ $<

$(LOADER): bpf_prog $(LIBBPF) $(HELPER_OBJECTS) $(USER_OBJECT)
	$(CC) -o $(LOADER) $(HELPER_OBJECTS) $(USER_OBJECT) $(LDLIBS)

all: $(LOADER)

.DEFAULT_GOAL := all
