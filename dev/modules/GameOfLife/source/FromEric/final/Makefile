# 
# Generic recursive Makefile
#

DIRS := $(dir $(shell find . -mindepth 2 -maxdepth 2 -name Makefile))

DIRS_CLEAN := $(addsuffix .clean,$(DIRS))

all:$(DIRS)

clean:$(DIRS_CLEAN)

$(DIRS):
	$(MAKE) -C $@

$(DIRS_CLEAN):
	$(MAKE) -C $(basename $@) clean

.PHONY:$(DIRS) $(DIRS_CLEAN) all clean
