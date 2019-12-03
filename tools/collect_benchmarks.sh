#!/bin/bash

# run me with cmake generated `make collect_benchmarks`

githash=`git log -n1 --pretty=%h`
timestamp=`date +%Y%m%d%H%M%S`
host=`uname -n`

build/src/run_benchmarks --benchmark_out_format=json --benchmark_out=benchmarks/${host}_${timestamp}_${githash}.json

