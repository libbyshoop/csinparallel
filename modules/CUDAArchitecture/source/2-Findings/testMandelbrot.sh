#! /bin/bash

for i in {1..512}
do
    for j in {1..512}
    do
        echo `./mandelbrot $i $j`
    done
done
