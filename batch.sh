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
LAT_LON="-48.56620427006758,-22.457495449468666,-47.9777042099919,-22.80261692472655"

# EXECUTIONS
for min_occ in "${MIN_OCCS[@]}"
do
	eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --date_start=1985-01-01 --date_end=2001-12-31 --min_occurrence=$min_occ --seasonal"
	eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --date_start=2002-01-01 --date_end=2018-12-31 --min_occurrence=$min_occ --seasonal"
done
############################################################################################