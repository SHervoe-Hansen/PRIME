#!/bin/bash

export AMBERHOME=/data9/stefan/amber24/bin
source /data9/stefan/amber24/amber.sh

# During volume relaxation the volume will change a lot causing an error on GPU code, therefore run CPU code
mpirun -np 64 pmemd.MPI -O -i equil.mdin -p ../0_preparation/insulin.explicit.parm7 -c ../2_heating/insulin.heat.rst7 -cpin ../0_preparation/insulin.cpin -o insulin.equil.mdout -r insulin.equil.rst7 -x insulin.equil.nc -cprestrt insulin.equil.cpin -cpout insulin.equil.cpout

