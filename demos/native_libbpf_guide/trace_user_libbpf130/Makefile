# SPDX-License-Identifier: GPL-2.0

CC     = $(CROSS_COMPILE)gcc
LD     = $(CROSS_COMPILE)ld
AS     = $(CROSS_COMPILE)as

HELPERS_PATH := ./helpers
TOOLS_PATH   := ./tools

CFLAGS += -iquote ./helpers/
CFLAGS += -iquote ./include/
CFLAGS += -I./tools/lib/
CFLAGS += -I./tools/include/
CFLAGS += -I./tools/include/uapi/

comma   := ,
dot-target = $(dir $@).$(notdir $@)
depfile = $(subst $(comma),_,$(dot-target).d)

LIBBPF   = $(TOOLS_PATH)/lib/bpf/libbpf.a
LDLIBS  += $(LIBBPF) -lelf -lz -lrt

SOURCES := $(wildcard *.c)

HELPER_OBJECTS := $(patsubst %.c,%.o,$(wildcard $(HELPERS_PATH)/*.c))
LOADER_OBJECT  := $(patsubst %.c,%,$(SOURCES))
USER_OBJECT    := $(patsubst %.c,%.o,$(SOURCES))
SKEL_OBJECT    := $(patsubst %.c,%.skel.h,$(SOURCES))
HEX_OBJECT     := $(patsubst %.c,%.hex,$(SOURCES))
BPF_OBJECT     := $(patsubst %.c,./progs/%.bpf.o,$(SOURCES))



.PHONY: clean

clean:
	rm -f *.ll *.o *.d .*.d *.hex *.skel.h $(LOADER_OBJECT) $(HELPERS_PATH)/*.o $(HELPERS_PATH)/.*.d
	make -C ./tools/lib/bpf/ clean
	make -C ./tools/build/feature clean
	make -C ./progs/ clean

$(LIBBPF):
	make -C ./tools/lib/bpf/

$(HELPER_OBJECTS): %.o:%.c
	$(CC) -Wp,-MD,$(depfile) $(CFLAGS)  -g -c -o $@ $<

$(BPF_OBJECT):./progs/%.bpf.o:./progs/%.bpf.c
	make -C ./progs/ BPF_TARGET=$(notdir $@)

$(HEX_OBJECT):%.hex:./progs/%.bpf.o
	hexdump -v -e '"\\\x" 1/1 "%02x"' ./progs/$(patsubst %.hex,%.bpf.o,$@)      > $@
	cp $@ hexdump.hex

$(SKEL_OBJECT):%.skel.h:%.hex
	printf 'size_t obj_buf_sz = '$(shell wc -c hexdump.hex | awk '{print $$1}')';\n' > $@
	printf 'char obj_buf[] = "'                                                     >> $@
	cat hexdump.hex | tr -d "\n"                                                    >> $@
	echo '";'                                                                       >> $@
	cp $@ skeleton.skel.h

$(USER_OBJECT):%.o:%.c %.skel.h
	$(CC) -Wp,-MD,$(depfile) $(CFLAGS)  -g -c -o $@ $<

$(LOADER_OBJECT): %:%.o
	$(CC) -g -o $@ $< $(HELPER_OBJECTS) $(LDLIBS)

all: $(LIBBPF) $(HELPER_OBJECTS) $(LOADER_OBJECT)
	@echo "Successfully remade target file 'all'."

.DEFAULT_GOAL := all
