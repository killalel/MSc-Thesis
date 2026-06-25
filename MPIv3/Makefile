CC = mpicc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

TARGET  = pcg_solver

PROBLEM ?= original

SRCS = main.c decomp.c poisson2d_$(PROBLEM).c linalg.c precond.c pcg.c multigrid.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

main.o: main.c decomp.h poisson2d.h linalg.h precond.h pcg.h
decomp.o: decomp.c decomp.h
poisson2d_$(PROBLEM).o: poisson2d_$(PROBLEM).c poisson2d.h decomp.h
linalg.o: linalg.c linalg.h decomp.h
precond.o: precond.c precond.h decomp.h
pcg.o: pcg.c pcg.h linalg.h precond.h decomp.h
multigrid.o: multigrid.c multigrid.h linalg.h decomp.h

clean:
	rm -f $(OBJS) $(TARGET)
