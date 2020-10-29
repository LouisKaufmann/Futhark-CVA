PARAMS='100000 100 [1,0.5] [10,20] [1,-0.5] 0.01 0.05 0.001 0.05'

TIME=/usr/bin/time

.PHONY: all
all: cva-c.out cva-o.out

.PHONY: test
test:
	futhark pyopencl --library cva.fut
	python3 parallelcva.py

cva-c.exe: cva.fut
	futhark c -o $@ $<

cva-o.exe: cva.fut
	futhark opencl -o $@ $<

# first make a heat-up execution
cva-c.out: cva-c.exe
	echo $(PARAMS) | ./$< > $@
	echo $(PARAMS) | $(TIME) ./$< > $@

# first make a heat-up execution
cva-o.out: cva-o.exe
	echo $(PARAMS) | ./$< > $@
	echo $(PARAMS) | $(TIME) ./$< > $@

.PHONY: clean
clean:
	rm -rf *~ *.exe *.c *.pyc __pycache__ cva.py runtime out.fut cva *.out
