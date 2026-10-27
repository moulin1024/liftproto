#!/usr/bin/env bash
# Full prototype run: lifting + certificates, mutation tests, compiler-parity bench.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf out
python3 run_lift.py fortran/cg.f
python3 mutate.py
cd bench
sed -n '/DOUBLE PRECISION FUNCTION DDOT/,/^      END/p;/SUBROUTINE DAXPY(/,/^      END/p' ../fortran/cg.f > legacy_kernels.f
echo "flags | ddot_unrolled ddot_plain daxpy_unrolled daxpy_plain [s] | bitwise_same_sum" > results.txt
for F in "-O0" "-O2" "-O3" "-O3 -march=native"; do
  gfortran -std=legacy -ffp-contract=off $F bench.f legacy_kernels.f plain_kernels.f -o b
  echo "$F | $(for r in 1 2 3; do ./b; done | sort -k1n | head -1)" | tee -a results.txt
done
