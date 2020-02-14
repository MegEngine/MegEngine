include ../../Makefile

MACHINE := $(shell $(MGB_CXX) -dumpmachine)

ifneq (, $(findstring android, $(MACHINE)))
    DEPS := $(MGB_LIB) ${MEGDNN_LIB}
    CXXFLAGS := $(MGB_CXXFLAGS) -fuse-ld=gold -Isrc
    LDFLAGS := -ldl -llog ${DEPS}
else
    DEPS := $(MGB_LIB) ${MEGDNN_LIB}
    CXXFLAGS := $(MGB_CXXFLAGS) -fuse-ld=gold -Isrc
    LDFLAGS := -ldl ${DEPS} ${MGB_LDFLAGS}
endif

TARGETS := load_and_run

all: $(TARGETS)

ifneq (,$(findstring gcov,$(MGB_LDFLAGS)))
	LDFLAGS += --coverage
endif

load_and_run: main.cpp src/* $(DEPS)
	$(MGB_CXX) -o $@ main.cpp src/*.cpp $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
