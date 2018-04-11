#!/bin/bash
# finding files in the PATH
IFS=:
for folder in $PATH; do
    echo "$folder:"
    for file in $folder/*; do
        if [ -x $file ]; then
            echo " $file"
        fi
    done
done
