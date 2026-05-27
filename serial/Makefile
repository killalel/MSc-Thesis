CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm
 
TARGET  = pcg_solver
SRCS = main.c poisson2d.c linalg.c precond.c pcg.c multigrid.c
OBJS    = $(SRCS:.c=.o)
 
.PHONY: all clean
 
all: $(TARGET)
 
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
 
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
 
# Header dependencies
main.o:     main.c poisson2d.h linalg.h precond.h pcg.h
poisson2d.o: poisson2d.c poisson2d.h
linalg.o:   linalg.c linalg.h
precond.o:  precond.c precond.h
pcg.o:      pcg.c pcg.h linalg.h precond.h
multigrid.o: multigrid.c multigrid.h linalg.h

clean:
	rm -f $(OBJS) $(TARGET)