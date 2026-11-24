#!/bin/bash

export AMBERHOME=/data9/stefan/amber24/bin
source /data9/stefan/amber24/amber.sh

pmemd.cuda -O -i min.mdin -p ../0_preparation/insulin.explicit.parm7 -c ../0_preparation/insulin.rst7 -o insulin.min.mdout -r insulin.min.rst7 -ref ../0_preparation/insulin.rst7 -cpin ../0_preparation/insulin.cpin

