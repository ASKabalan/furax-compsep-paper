#!/bin/bash

# Corrected Bash loop over BS values
for BS in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000
do
    echo "Running KMeans with BS = $BS"
    python 08-KMeans-model.py -n 64 -ns 100 -nr 1.0 -pc "$BS" 500 500 -tag c1d1s1 -m GAL020 -i LiteBIRD
    # Zone 2 mask of GAL040 - GAL020
    python 08-KMeans-model.py -n 64 -ns 100 -nr 1.0 -pc "$BS" 500 500 -tag c1d1s1 -m GAL040 -i LiteBIRD
    # Zone 3 mask of GAL060 - GAL040
    python 08-KMeans-model.py -n 64 -ns 100 -nr 1.0 -pc "$BS" 500 500 -tag c1d1s1 -m GAL060 -i LiteBIRD
done
