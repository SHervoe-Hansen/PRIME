#!/bin/bash
export AMBERHOME=/data9/stefan/amber24/bin
source /data9/stefan/amber24/amber.sh

pmemd.cuda -O -i min.mdin -p ../0_preparation/Insulin_fibril_1.explicit.parm7 -c ../0_preparation/Insulin_fibril_1.rst7 -o Insulin_fibril_1.min.mdout -r Insulin_fibril_1.min.rst7 -ref ../0_preparation/Insulin_fibril_1.rst7 -cpin ../0_preparation/Insulin_fibril_1.cpin

