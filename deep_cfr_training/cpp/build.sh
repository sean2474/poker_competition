#!/bin/bash
# Build C++ traversal library for Python ctypes
# Usage: bash build.sh
set -e
cd "$(dirname "$0")"
if [[ "$OSTYPE" == "darwin"* ]]; then
    clang++ -O3 -shared -fPIC -std=c++17 -o libtraversal.dylib traversal.cpp -lpthread
    echo "Built libtraversal.dylib ($(du -h libtraversal.dylib | cut -f1))"
else
    g++ -O3 -shared -fPIC -std=c++17 -o libtraversal.so traversal.cpp -lpthread -fopenmp
    echo "Built libtraversal.so ($(du -h libtraversal.so | cut -f1))"
fi
