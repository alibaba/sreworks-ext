# SPDX-License-Identifier: GPL-2.0

ifeq ($(srctree),)
srctree := $(patsubst %/,%,$(dir $(CURDIR)))
srctree := $(patsubst %/,%,$(dir $(srctree)))
srctree := $(patsubst %/,%,$(dir $(srctree)))
$(info Determined 'srctree' to be $(srctree))
endif

define allow-override
  $(if $(or $(findstring environment,$(origin $(1))),\
            $(findstring command line,$(origin $(1)))),,\
    $(eval $(1) = $(2)))
endef

# Allow setting CC and AR, or setting CROSS_COMPILE as a prefix.
$(call allow-override,CC,$(CROSS_COMPILE)gcc)
$(call allow-override,AR,$(CROSS_COMPILE)ar)

include $(srctree)/tools/scripts/Makefile.arch

#
## Include saner warnings here, which can catch bugs:
#
EXTRA_WARNINGS := -Wbad-function-cast 
EXTRA_WARNINGS += -Wdeclaration-after-statement 
EXTRA_WARNINGS += -Wformat-security 
EXTRA_WARNINGS += -Wformat-y2k 
EXTRA_WARNINGS += -Winit-self 
EXTRA_WARNINGS += -Wmissing-declarations 
EXTRA_WARNINGS += -Wmissing-prototypes 
EXTRA_WARNINGS += -Wnested-externs 
EXTRA_WARNINGS += -Wno-system-headers 
EXTRA_WARNINGS += -Wold-style-definition 
EXTRA_WARNINGS += -Wpacked 
EXTRA_WARNINGS += -Wredundant-decls 
EXTRA_WARNINGS += -Wshadow 
EXTRA_WARNINGS += -Wstrict-prototypes 
EXTRA_WARNINGS += -Wswitch-default 
EXTRA_WARNINGS += -Wswitch-enum 
EXTRA_WARNINGS += -Wundef 
EXTRA_WARNINGS += -Wwrite-strings 
EXTRA_WARNINGS += -Wformat 
EXTRA_WARNINGS += -Wstrict-aliasing=3 
EXTRA_WARNINGS += -fno-strict-aliasing
EXTRA_WARNINGS += -Wno-switch-enum
EXTRA_WARNINGS += -Wno-type-limits

CC_NO_CLANG := $(shell $(CC) -dM -E -x c /dev/null | grep -Fq "__clang__"; echo $$?)

define allow-override
  $(if $(or $(findstring environment,$(origin $(1))),\
            $(findstring command line,$(origin $(1)))),,\
    $(eval $(1) = $(2)))
endef

# Allow setting various cross-compile vars or setting CROSS_COMPILE as a prefix.
$(call allow-override,CC,$(CROSS_COMPILE)gcc)
$(call allow-override,AR,$(CROSS_COMPILE)ar)
$(call allow-override,LD,$(CROSS_COMPILE)ld)
$(call allow-override,CXX,$(CROSS_COMPILE)g++)
$(call allow-override,STRIP,$(CROSS_COMPILE)strip)

ifeq ($(CC_NO_CLANG), 1)
EXTRA_WARNINGS += -Wstrict-aliasing=3
endif

ifneq ($(filter 3.%,$(MAKE_VERSION)),)  # make-3
EXTRA_WARNINGS += -fno-strict-aliasing
endif

FEATURE_TESTS = libelf bpf reallocarray

INCLUDES = -I. -I$(srctree)/tools/include -I$(srctree)/tools/include/uapi -I$(srctree)/tools/perf
FEATURE_CHECK_CFLAGS-bpf = $(INCLUDES)



#### feature test begin
feature_dir := $(srctree)/tools/build/feature

feature_check = $(eval $(feature_check_code))
define feature_check_code
  cmd-$(1) := $(shell echo $(MAKE) OUTPUT=$(OUTPUT_FEATURES) CFLAGS="$(EXTRA_CFLAGS) $(FEATURE_CHECK_CFLAGS-$(1))" CXXFLAGS="$(EXTRA_CXXFLAGS) $(FEATURE_CHECK_CXXFLAGS-$(1))" LDFLAGS="$(LDFLAGS) $(FEATURE_CHECK_LDFLAGS-$(1))" -C $(feature_dir) $(OUTPUT_FEATURES)test-$1.bin)
  feature-$(1) := $(shell $(MAKE) OUTPUT=$(OUTPUT_FEATURES) CFLAGS="$(EXTRA_CFLAGS) $(FEATURE_CHECK_CFLAGS-$(1))" CXXFLAGS="$(EXTRA_CXXFLAGS) $(FEATURE_CHECK_CXXFLAGS-$(1))" LDFLAGS="$(LDFLAGS) $(FEATURE_CHECK_LDFLAGS-$(1))" -C $(feature_dir) $(OUTPUT_FEATURES)test-$1.bin >/dev/null 2>/dev/null && echo 1 || echo 0)
endef

set_test_all_flags = $(eval $(set_test_all_flags_code))
define set_test_all_flags_code
  FEATURE_CHECK_CFLAGS-all  += $(FEATURE_CHECK_CFLAGS-$(1))
  FEATURE_CHECK_LDFLAGS-all += $(FEATURE_CHECK_LDFLAGS-$(1))
endef

$(foreach feat,$(FEATURE_TESTS),$(call set_test_all_flags,$(feat)))

$(foreach feat,$(FEATURE_TESTS),$(call feature_check,$(feat)))
#$(call feature_check,libelf)
#$(call feature_check,libelf-mmap)
#$(call feature_check,bpf)
#$(call feature_check,reallocarray)

$(info cmd-libelf---$(cmd-libelf))
$(info cmd-bpf---$(cmd-bpf))
$(info cmd-libelf-mmap---$(cmd-libelf-mmap))
$(info cmd-reallocarray---$(cmd-reallocarray))

$(info libelf---$(feature-libelf))
$(info bpf---$(feature-bpf))
$(info libelf-mmap---$(feature-libelf-mmap))
$(info reallocarray---$(feature-reallocarray))

#### feature test end

comma   := ,

dot-target = $(dir $@).$(notdir $@)
depfile = $(subst $(comma),_,$(dot-target).d)

# Set compile option CFLAGS
ifdef EXTRA_CFLAGS
  CFLAGS := $(EXTRA_CFLAGS)
else
  CFLAGS := -g
endif

ifeq ($(feature-libelf-mmap), 1)
  override CFLAGS += -DHAVE_LIBELF_MMAP_SUPPORT
endif
ifeq ($(feature-reallocarray), 0)
  override CFLAGS += -DCOMPAT_NEED_REALLOCARRAY
endif

# Append required CFLAGS
override CFLAGS += $(EXTRA_WARNINGS)
override CFLAGS += -Werror -Wall
override CFLAGS += -O2
override CFLAGS += $(INCLUDES)
override CFLAGS += -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64

objects = $(patsubst %.c,%.o,$(wildcard *.c))

$(objects): %.o:%.c
	$(CC) -Wp,-MD,$(depfile) -Wp,-MT,$@ $(CFLAGS) -D"BUILD_STR(s)=#s" -o $@ -c $<

precheck: force elfdep bpfdep

libbpf-in.o: precheck $(objects)
	ld -r -o libbpf-in.o $(objects)

all: libbpf-in.o
	$(AR) rcs libbpf.a libbpf-in.o

### Cleaning rules

clean:
	rm -f *.o *.d .*.d libbpf.a libbpf-in.o 

PHONY += force elfdep bpfdep precheck
force:

elfdep:
	@if [ "$(feature-libelf)" != "1" ]; then echo "No libelf found"; exit 1 ; fi

bpfdep:
	@if [ "$(feature-bpf)" != "1" ]; then echo "BPF API too old"; exit 1 ; fi

.PHONY: $(PHONY)
.DEFAULT_GOAL := all
