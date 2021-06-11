#!/usr/bin/python
# -*- coding: utf-8 -*-

##################################################################################################################
# ### Miscellaneous
# ### Module responsible for storing extra data processing functions, accuracy measures and others.
##################################################################################################################

# Dependents Modules
import pandas as pd
import gc
import numpy as np

# Remove duplicated dates
def remove_duplicated_dates(dates: list):
  visited = []
  for i,date in enumerate(dates):
    if date.strftime("%Y-%m-%d") in visited:
      del dates[i]
    else:
      visited.append(date.strftime("%Y-%m-%d"))
  return dates