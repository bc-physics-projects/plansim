#!/bin/bash

maturin build --release --pgo --features polars/performant --auditwheel=repair