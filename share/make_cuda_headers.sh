#!/bin/sh

for f in $(find -L redGrapes/helpers/cupla -name '*.hpp');
do
    TARGET=$(echo $f | sed -e 's/cupla/cuda/g');
    mv -f $TARGET $TARGET.bak;
    sed -e 's/Cupla/Cuda/g' -e 's/cupla/cuda/g' $f > $TARGET;
done

