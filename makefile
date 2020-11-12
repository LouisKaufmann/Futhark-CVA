PARAMS='10000 100 [1,0.5,0.25,0.1,0.3,0.1,2,3,1] [10,20,5,20,5,10,10,20,50] [1,1,1,1,1,1,1,1,1] 0.01 0.05 0.001 0.05'

TIME=/usr/bin/time

.PHONY: all
all: cva-expand.out cva-map.out

cva-c.exe: cva-map.fut
	futhark c -o $@ $<

cva-map.exe: cva-map.fut
	futhark opencl -o $@ $<

cva-expand.exe: cva-expand.fut
	futhark opencl -o $@ $<

cva-c.out: cva-c.exe
	echo $(PARAMS) | ./$< > $@
	echo $(PARAMS) | $(TIME) ./$< > $@

cva-map.out: cva-map.exe
	echo $(PARAMS) | ./$< > $@
	echo $(PARAMS) | $(TIME) ./$< > $@

cva-expand.out: cva-expand.exe
	echo $(PARAMS) | ./$< > $@
	echo $(PARAMS) | $(TIME) ./$< > $@

.PHONY: clean
clean:
	rm -rf *~ *.exe *.c *.pyc __pycache__ cva.py runtime out.fut cva-map cva-expand *.out
