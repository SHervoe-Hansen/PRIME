#!/bin/bash

export AMBERHOME=/data9/stefan/amber24
source /data9/stefan/amber24/amber.sh

# During volume relaxation the volume will change a lot causing an error on GPU code, therefore run CPU code
mpirun -np 64 pmemd.MPI -O -i equil.mdin -p ../0_preparation/Insulin_fibril_1.explicit.parm7 -c ../2_heating/Insulin_fibril_1.heat.rst7 -ref ../2_heating/Insulin_fibril_1.heat.rst7 -cpin ../0_preparation/Insulin_fibril_1.cpin -o Insulin_fibril_1.equil.mdout -r Insulin_fibril_1.equil.rst7 -x Insulin_fibril_1.equil.nc -cprestrt Insulin_fibril_1.equil.cpin -cpout Insulin_fibril_1.equil.cpout

