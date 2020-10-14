test:
	futhark pyopencl --library cva.fut
	python parallelcva.py