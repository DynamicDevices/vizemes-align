# Host-side runtime: mel frontend + stub (default) or ONNX Runtime (ORT_ROOT set).
#
# Stub:
#   make
# ORT (nix develop .#train exports ORT_ROOT, or release tarball):
#   make ORT_ROOT=/path/to/onnxruntime-linux-x64-1.20.1
#   make ORT_ROOT=... smoke

CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -O2
INCLUDES = -Iinclude -Ifrontends/mel
LDFLAGS ?=
LDLIBS = -lm

MEL_OBJS = frontends/mel/mel_spectrogram.o src/mel_frontend.o

ifdef ORT_ROOT
  INCLUDES += -I$(ORT_ROOT)/include
  LDFLAGS += -L$(ORT_ROOT)/lib -Wl,-rpath,$(ORT_ROOT)/lib
  LDLIBS += -lonnxruntime
  RUNTIME_OBJS = src/viseme_runtime.o src/sidecar_json.o
  CFLAGS += -DVIZEMES_WITH_ORT=1
else
  RUNTIME_OBJS = src/viseme_runtime_stub.o
endif

OBJS = $(MEL_OBJS) $(RUNTIME_OBJS)

.PHONY: all clean smoke

all: build/libvizemes_runtime.a

build/libvizemes_runtime.a: $(OBJS)
	mkdir -p build
	ar rcs $@ $(OBJS)

build/smoke_context: tools/smoke_context.c build/libvizemes_runtime.a
ifndef ORT_ROOT
	$(error smoke requires ORT_ROOT (nix develop .#train or onnxruntime release prefix))
endif
	mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/smoke_context.c -Lbuild -lvizemes_runtime $(LDFLAGS) $(LDLIBS)

smoke: build/smoke_context
	python3 tools/dump_demo_row.py >/tmp/vizemes-dump-row.txt
	# Match C softmax-argmax to Python ORT (labels on ci-smoke are weak).
	PRED=$$(python3 tools/ort_pred_row.py); \
	./build/smoke_context ../export/ci-smoke/model.json ../export/ci-smoke/model.onnx \
		../export/ci-smoke/demo_row0.f32 $$PRED

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(MEL_OBJS) src/viseme_runtime.o src/viseme_runtime_stub.o src/sidecar_json.o \
		build/libvizemes_runtime.a build/smoke_context
