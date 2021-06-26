#!/usr/bin/env python
# coding: utf-8

# # Filter Itemid Prescriptions
# 
# First, we find the number of observation that has each unit of measurement. If the top frequent one has the observations less than 90% of total, then we discard it.
# 
# Also, there are many medicine that never be used by any patients, so we also discard it.

# In[1]:


from __future__ import print_function

import psycopg2
import datetime
import sys
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import itertools
import os.path
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool, cpu_count

from utils import getConnection

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


num_workers = cpu_count() // 2
res_dir = '/media/data/mimic/res'


# In[3]:


conn = getConnection()
    
_adm = np.load(f'{res_dir}/admission_ids.npy', allow_pickle=True).tolist()
admission_ids = _adm['admission_ids']
admission_ids_txt = _adm['admission_ids_txt']

db = np.load(f'{res_dir}/itemids.npy', allow_pickle=True).tolist()
input_itemid = db['input']
output_itemid = db['output']
chart_itemid = db['chart']
lab_itemid = db['lab']
microbio_itemid = db['microbio']
prescript_itemid = db['prescript']


# In[4]:


def stat_prescript_unit_task(i, admission_ids_txt):
    conn = getConnection()
    # foreach medicine, list the dose unit of medicine and the count of observations that use that unit.
    cur = conn.cursor()
    cur.execute('SELECT dose_unit_rx, count(dose_unit_rx) FROM mimiciii.prescriptions WHERE formulary_drug_cd = \''+ str(i) +'\' and hadm_id in (select * from admission_ids) group by dose_unit_rx')
    outputunits = cur.fetchall()
    
    # sort it descendently
    outputunits = sorted(outputunits, key=lambda tup: tup[1])
    outputunits.reverse()
    return (i, outputunits)

p = Pool(num_workers)
results = [p.apply_async(stat_prescript_unit_task, args=(i, admission_ids_txt)) for i in prescript_itemid]
p.close()
p.join()

results = [x.get() for x in results]
np.save(f'{res_dir}/filtered_prescript_raw.npy', {'raw': results})
print('saved!')


# In[5]:


valid_prescript =[]
valid_prescript_unit =[]
dropped_id = []
notfound = []
results = np.load(f'{res_dir}/filtered_prescript_raw.npy', allow_pickle=True).tolist()['raw']

for x in results:
    i, outputunits = x[0], x[1]
    # check if medicine is never used by anybody, then discard it.
    total = 0
    for o in outputunits:
        total += o[1]
    if(total == 0 ): 
#         print("\n**PRES DROPPED "+str(i))
        notfound.append(i)
        for o in outputunits:
            print(o)
        print('')
        continue
        
    # calculate the percentage of observation of main unit.
    percentage = float(outputunits[0][1]) / total *100.
    if(percentage < 90): 
        # never drop the list A
#         if(i in manual_valid):
#             print("\n\n****PRES NOT DROPPED "+str(i) + " : " + "{:.2f}".format(percentage) + " : " + str(len(outputunits))+" : "+ str(outputunits)+"\n")
        
        # drop it !
#         else:
            dropped_id.append(i)
            print("\n\n****PRES DROPPED "+str(i) + " : " + "{:.2f}".format(percentage) + " : " + str(len(outputunits))+" : "+ str(outputunits)+"\n")
            continue
#     print("PRES "+str(i) + " : " + "{:.2f}".format(percentage) + " : " + str(len(outputunits))+" : "+ str(outputunits))
    
    # keep it and also save the main unit of it.
    valid_prescript.append(i)
    valid_prescript_unit.append(outputunits[0][0])


# In[6]:


print(valid_prescript, valid_prescript_unit)
np.save(f'{res_dir}/filtered_prescript.npy',{'id':valid_prescript,'unit':valid_prescript_unit})
print('saved!')

