#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"
CLEAR="sudo pkill -f /home/pedro/anaconda3"

# ATTRIBUTES
declare -a MIN_OCCS=(3 4)

# SHOW BASE DIR
echo "$PYTHON $BASEDIR/$SCRIPT"


############################################################################################
## PERIOD 1
LAT_LON="-48.84725671390528,-22.04547298853004,-47.71712046185493,-23.21347463046867"

# EXECUTIONS
for min_occ in "${MIN_OCCS[@]}"
do
	eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --date_start=1985-01-01 --date_end=2001-12-31 --min_occurrence=$min_occ"
done

############################################################################################


############################################################################################
## PERIOD 2
LAT_LON="-48.84725671390528,-22.04547298853004,-47.71712046185493,-23.21347463046867"

# EXECUTIONS
for min_occ in "${MIN_OCCS[@]}"
do
	eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --date_start=2002-01-01 --date_end=2018-12-31 --min_occurrence=$min_occ"
done

############################################################################################