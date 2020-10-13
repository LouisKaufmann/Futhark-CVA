test:
	futhark pyopencl --library cva.fut
	python parrallelcva.py