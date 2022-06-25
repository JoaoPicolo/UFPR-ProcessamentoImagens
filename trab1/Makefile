CC     = g++
CFLAGS = -Wall
LFLAGS = 

PROG = histograma 
OBJS = 

.PHONY: clean purge all

all: histograma

%.o: %.cpp %.hpp
	$(CC) -c $(CFLAGS) $<

$(PROG) : % :  $(OBJS) %.o
	$(CC) -o $@ $^ $(LFLAGS)

purge:
	@rm -f *~ *.bak

clean: 
	@rm -f *.o core a.out
	@rm -f $(PROG)