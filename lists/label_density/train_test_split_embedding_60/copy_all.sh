#!/bin/bash

declare -a arr=("0.5" "1" "1.5" "2" "2.5" "5" "10" "15" "20" "25")
for i in "${arr[@]}"
do
	cp *testing* ../train_test_split_embedding_"$i"_percent/
	cp *validation* ../train_test_split_embedding_"$i"_percent/
done
