accel_c.so: accel_c.c
	gcc -shared -fPIC -O4 -fopenmp --std=c99 accel_c.c -o accel_c.so
