#!/bin/bash

export AMBERHOME=/data9/stefan/amber24
source /data9/stefan/amber24/amber.sh

pmemd.cuda -O -i heat.mdin -c ../1_minimization/Insulin_fibril_1.min.rst7 -p ../0_preparation/Insulin_fibril_1.explicit.parm7 -ref ../1_minimization/Insulin_fibril_1.min.rst7 -cpin ../0_preparation/Insulin_fibril_1.cpin -o Insulin_fibril_1.heat.mdout -r Insulin_fibril_1.heat.rst7 -x Insulin_fibril_1.heat.nc
