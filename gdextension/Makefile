# Host-side smoke (no Godot yet): compile mel frontend + stub runtime.
CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -O2
INCLUDES = -Iinclude -Ifrontends/mel

OBJS = frontends/mel/mel_spectrogram.o src/mel_frontend.o src/viseme_runtime_stub.o

.PHONY: all clean

all: build/libvizemes_runtime.a

build/libvizemes_runtime.a: $(OBJS)
	mkdir -p build
	ar rcs $@ $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJS) build/libvizemes_runtime.a
