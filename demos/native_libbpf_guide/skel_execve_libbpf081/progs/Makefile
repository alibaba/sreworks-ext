# SPDX-License-Identifier: GPL-2.0

LLC          ?= llc
CLANG        ?= clang
OPT          ?= opt
LLVM_DIS     ?= llvm-dis
LLVM_OBJCOPY ?= llvm-objcopy
BTF_PAHOLE   ?= pahole
ROOTDIR      ?= ./../

HELPERS_PATH := $(ROOTDIR)/helpers
HELPERINCLUDE = -I$(HELPERS_PATH)

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


try-run = $(shell set -e;if ($(1)) >/dev/null 2>&1;then echo "$(2)";else echo "$(3)";fi)
cc-option-yn = $(call try-run, $(CC) -Werror $(KBUILD_CFLAGS) $(1) -x c -c /dev/null -o /dev/null,y,n)

include $(KERNEL-DEVEL)/include/config/auto.conf

ifdef CONFIG_HAVE_FENTRY
  ifeq ($(call cc-option-yn, -mfentry),y)
    CC_FLAGS_USING      += -DCC_USING_FENTRY
  endif
endif

NOWARN += -Wno-unused-value
NOWARN += -Wno-pointer-sign
NOWARN += -Wno-compare-distinct-pointer-types
NOWARN += -Wno-gnu-variable-sized-type-not-at-end
NOWARN += -Wno-address-of-packed-member
NOWARN += -Wno-tautological-compare
NOWARN += -Wno-unknown-warning-option

NOFLAG += -fno-asynchronous-unwind-tables
NOFLAG += -fno-stack-protector

CLANG_VERSION=$(shell echo | clang -dM -E - | grep __clang_major__ | awk '{print $$3}')

ifeq ($(shell expr $(CLANG_VERSION) '<' 11), 1)
  ASM_GOTO_FLAG = -include asm_goto_workaround.h
endif

.PHONY: clean $(CLANG) $(LLC)

clean:
	rm -f *.ll *.o *.d .*.d

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
	$(CLANG) -iquote ./../include/ -I./../tools/lib/ $(LINUXINCLUDE) $(HELPERINCLUDE) $(NOSTDINC_FLAGS) -D__KERNEL__ -D__BPF_TRACING__ -D__TARGET_ARCH_$(ARCH) $(CC_FLAGS_USING) $(NOWARN) $(NOFLAG)  $(ASM_GOTO_FLAG) -O2 -emit-llvm -Xclang -disable-llvm-passes -c $< -o - | \
        $(OPT) -O2 -mtriple=bpf-pc-linux | \
        $(LLVM_DIS) | \
        $(LLC) -march=bpf $(LLC_FLAGS) -filetype=obj -o $@

all: $(VERIFY) $(BPF_OBJECT)

.DEFAULT_GOAL := all
