import os

# The training loop multiplies many small matrices; with several BLAS threads per process it ran 2-3x slower here
# (and the parallel training runs fought over the cores). One thread per process must be set before NumPy is imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
