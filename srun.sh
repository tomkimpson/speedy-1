#!/bin/bash

# $1 = experiment name (eg ElNino10s11e)
# $2 = experiment no. for restart file (i.e. initial condition, 0 = from rest)
# $3 = choice of SST anomaly (0 = no forcing, 1 = El Nino (static, scalefactor = 4, no mean adjust))
# $4 = number of significand bits (52 = double, 23 = single, 10 = half)
# $5 = stochastic rounding (sr_true or sr_false)

set -e

if (($# != 5)); then
	echo 'Usage: '${0}' experiment name, restart no., choice of SST anomaly forcing (0 = none, 1=El Nino), number of significand bits (52 = double, 23 = single, 10 = half), with or without stochastic rounding (sr_true or sr_false)' 1>&2
    exit 1
fi

# Define directory names
UT=`pwd -P`
OUT=/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/$1
INP=${UT}/initial_conditions/exp_${2}
mkdir -p ${OUT}

# Setup files
executable=${UT}/source/imp.exe
echo "Using restart namelist"
namelist=${UT}/setup/speedy10year.nml
output=${UT}/setup/default_outputs.nml
precisions=${UT}/setup/${4}sig11exp.nml

# Copy files from basic version directory
mkdir -p ${OUT}
find ${OUT} -type f -delete
find ${OUT} -type l -delete
find ${OUT} -mindepth 1 -type d -delete
cp ${executable} ${OUT}/imp.exe
cp ${namelist}   ${OUT}/speedy.nml
cp ${output}     ${OUT}/output_requests.nml
cp ${precisions} ${OUT}/precisions.nml

# Link restart file (i.e. set initial conditions)
cp ${INP}/*.rst ${OUT}

# Link input files
BC=${UT}/data/bc/t30
SH=${UT}/hflux

cd ${OUT}
ln -s ${BC}/climatology.nc climatology.nc

#set the SST anomaly forcing
if (($3 == 0)); then
	ln -s ${BC}/blank_anomalies.nc   anomalies.nc
elif (($3 == 1)); then
	ln -s ${BC}/elNinoSSTanomaly_static_nocutoff_scalefactor4_no-mean-adjust.nc anomalies.nc
fi

ln -s ${SH}/hflux_speedy_ver41_1979_2008_clim.grd fort.31

# Link netCDF library
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/share/netcdf/lib
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${UT}/../rpe_complex/lib/

if [[ $5 == sr_true ]] 
then
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${UT}/../rpe_complex_stochastic/lib/
	echo "stochastic rounding is on"
elif [[ $5 == sr_false ]]
then
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${UT}/../rpe_complex_stochastic/lib/
	echo "stochastic rounding is off"
else
	echo "usage: final argument was $5 but must be one of sr_true or sr_false"
	exit 1
fi

echo "I will now proceed to run SPEEDY"

time ./imp.exe | tee out.lis
