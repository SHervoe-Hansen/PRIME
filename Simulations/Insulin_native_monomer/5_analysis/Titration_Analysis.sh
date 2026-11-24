#!/bin/sh

cpin=../0_preparation/insulin.cpin

rm reordered_cpouts.pH_* pH* titration_curves.dat
echo cphstats --fix-remd reordered_cpouts ../4_production/run.001/pH*/insulin.pHREMD.cpout
cphstats --fix-remd reordered_cpouts ../4_production/run.001/pH*/insulin.pHREMD.cpout

echo "# 1    2" >titration_curves.dat
echo "# pH   Residue" >>titration_curves.dat

for pH in $(seq 1.8 0.2 14)
do
        pH_fmt=$(printf "%4.2f" $pH)
        cphstats -i $cpin reordered_cpouts.pH_${pH_fmt} -o pH${pH_fmt}_calcpka.dat --population pH${pH_fmt}_populations.dat

        echo $pH_fmt | awk '{printf("%4.2f ",$1)}'>> titration_curves.dat
        cat pH${pH_fmt}_calcpka.dat | awk '/Frac Prot/ {printf("%5.3f ",1-$10)}' >>titration_curves.dat
        echo $pH_fmt | awk '{printf(" \n",$1)}'>> titration_curves.dat
done
