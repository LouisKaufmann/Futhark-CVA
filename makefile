PARAMS='10000 100 [1,0.5,0.25,0.1,0.3,0.1,2,3,1] [10,20,5,5,50,20,30,15,18] [1,-0.5,1,1,1,1,1,1,1] 0.01 0.05 0.001 0.05'

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

cva-dyn.exe: cvadynmem.fut
	futhark opencl -o $@ $<

cva-c.out: cva-c.exe
	echo $(PARAMS) | ./$< > $@
	echo $(PARAMS) | $(TIME) ./$< > $@

cva-o.out: cva-o.exe
	echo $(PARAMS) | ./$< > $@
	echo $(PARAMS) | $(TIME) ./$< > $@

cva-dyn.out: cva-dyn.out
	echo $(PARAMS) | ./$< > $@
	echo $(PARAMS) | $(TIME) ./$< > $@

.PHONY: clean
clean:
	rm -rf *~ *.exe *.c *.pyc __pycache__ cva.py runtime out.fut cva cva2 cvaold *.out
