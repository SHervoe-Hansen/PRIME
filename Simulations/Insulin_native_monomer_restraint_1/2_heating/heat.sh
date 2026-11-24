#!/bin/bash

export AMBERHOME=/data9/stefan/amber24/bin
source /data9/stefan/amber24/amber.sh

pmemd.cuda -O -i heat.mdin -c ../1_minimization/insulin.min.rst7 -p ../0_preparation/insulin.explicit.parm7 -cpin ../0_preparation/insulin.cpin -o insulin.heat.mdout -r insulin.heat.rst7 -x insulin.heat.nc


