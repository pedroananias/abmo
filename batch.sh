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
for year in {1985..2018}
do
	for min_occ in "${MIN_OCCS[@]}"
	do
		eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --date_start=$year-01-01 --date_end=$year-12-31 --min_occurrence=$min_occ"
	done
done
############################################################################################