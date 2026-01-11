#!/bin/bash

# current script path
CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


bash $CURR_DIR/06-validate_furax.sh 1.0 c1d1s1 adam
bash $CURR_DIR/06-validate_furax.sh 0.0 c1d1s1 adam
bash $CURR_DIR/06-validate_furax.sh 1.0 c1d0s0 adam
bash $CURR_DIR/06-validate_furax.sh 0.0 c1d0s0 adam

bash $CURR_DIR/06-validate_furax.sh 1.0 c1d1s1 active_set
bash $CURR_DIR/06-validate_furax.sh 0.0 c1d1s1 active_set
bash $CURR_DIR/06-validate_furax.sh 1.0 c1d0s0 active_set
bash $CURR_DIR/06-validate_furax.sh 0.0 c1d0s0 active_set

bash $CURR_DIR/06-validate_furax.sh 1.0 c1d1s1 scipy_tnc
bash $CURR_DIR/06-validate_furax.sh 0.0 c1d1s1 scipy_tnc
bash $CURR_DIR/06-validate_furax.sh 1.0 c1d0s0 scipy_tnc
bash $CURR_DIR/06-validate_furax.sh 0.0 c1d0s0 scipy_tnc

bash $CURR_DIR/07-validate_fg.sh 1.0 c1d1s1
bash $CURR_DIR/07-validate_fg.sh 1.0 c1d1s1
bash $CURR_DIR/07-validate_fg.sh 1.0 c1d0s0
bash $CURR_DIR/07-validate_fg.sh 0.0 c1d0s0
