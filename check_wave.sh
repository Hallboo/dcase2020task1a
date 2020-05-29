#!/bin/bash

wave_folder=$1

[ -d $wave_folder ]||exit 1;

the_sample=$(find $wave_folder -name "*.wav" | head -1)
echo $the_sample
sox -V $the_sample -n
