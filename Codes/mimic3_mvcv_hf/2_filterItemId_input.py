#!/usr/bin/env python
# coding: utf-8

# # Filter Itemid Input
# 
# This script is used for filtering itemids from TABLE INPUTEVENTS.
# 
# 1. We check number of units of each itemid and choose the major unit as the target of unit conversion.
# 2. In this step we do not apply any filtering to the data.
# 
# ## Output
# 
# 1. itemid of observations for inputevents.
# 2. unit of measurement for each itemid.

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

# %matplotlib inline


# In[2]:


num_workers = cpu_count() // 2


# In[ ]:


res_dir = "/media/data/mimic/res"


# In[3]:


try:
    conn = getConnection()
    print('Connected to Postgre Database!')
except:
    print('Fail to connect!')
    
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


valid_input = []
valid_input_unit = []


# Compare the speed between two ways of query. In order to accelerate, we manually create a TABLE ADMISSION_IDS to store all admission_ids.

# In[5]:


cur = conn.cursor()
start = datetime.datetime.now()
for t in range(10):
    cur = conn.cursor()
    cur.execute('select coalesce(amountuom, \'\'), count(*) from mimiciii.inputevents_cv where itemid=30044 and hadm_id in (select * from admission_ids) group by amountuom')
    res = cur.fetchall()
end = datetime.datetime.now()
print(end - start)
print(res)

start = datetime.datetime.now()
for t in range(10):
    cur = conn.cursor()
    cur.execute('select coalesce(amountuom, \'\'), count(*) from mimiciii.inputevents_cv where itemid=30044 and hadm_id in ({0}) group by amountuom'.format(admission_ids_txt))
    res = cur.fetchall()
end = datetime.datetime.now()
print(end - start)
print(res)


# In[6]:


# inputevents
def stat_inputevents_unit_task(itemid, admission_ids_txt):
    tconn = getConnection()
    tcur = tconn.cursor()
#     tcur.execute('SELECT amountuom, count(amountuom) FROM mimiciii.inputevents_cv \
#                 WHERE amountuom is not null and itemid = '+ str(itemid) +' and hadm_id in ('+admission_ids_txt+') group by amountuom')
#     tcur.execute('select coalesce(amountuom, \'\'), count(*) from (select amountuom, itemid, hadm_id from mimiciii.inputevents_cv union select amountuom, itemid, hadm_id from mimiciii.inputevents_mv) \
#         where itemid={0} and hadm_id in (select hadm_id from admission_ids) group by amountuom'.format(itemid))
    tcur.execute('select amountuom, sum(count::int) from (                    select coalesce(amountuom, \'\') as amountuom, count(*) from mimiciii.inputevents_cv where itemid = {0} and hadm_id in (select * from admission_ids) group by amountuom                    union all                    select coalesce(amountuom, \'\') as amountuom, count(*) from mimiciii.inputevents_mv where itemid = {0} and hadm_id in (select * from admission_ids) group by amountuom                    ) as t where amountuom<>\'\' group by amountuom'.format(itemid))
    outputunits = tcur.fetchall()
    outputunits = sorted(outputunits, key=lambda tup: tup[1])
    outputunits.reverse()
    total = 0
    for o in outputunits:
        total += o[1]
    if(total == 0 ):
        return (itemid, None, None)
    percentage = float(outputunits[0][1]) / total *100.0
    tconn.close()
    return (itemid, percentage, outputunits)

p = Pool(num_workers)
valid_vupairs = [p.apply_async(stat_inputevents_unit_task, args=(i, admission_ids_txt)) for i in input_itemid]
p.close()
p.join()
valid_vupairs = [x.get() for x in valid_vupairs]


# ## iterate thru each itemID
# For each item id, we count number of observations for each unit of measurement.
# 
# For example,
# IN 225883 : 98.24 : 3 : [('dose', 16477L), ('mg', 251L), ('grams', 44L)]
# This means that for itemid 225883, there are:
# 1. 16477 records using dose as its unit of measurement.
# 2. 251 records using mg as its unit of measurement.
# 3. 44 records using grams as its unit of measurement.
# 
# dose has 98.24% over all the observations for this itemid, we can say that dose is a majority unit. 
# 1. We will keep this itemid because 98% is high. we can relatively safe to discard the observations that has different unit of measurement. i.e. if we discard mg and grams, we lose 251+44 records which is little, compared to 16477 records we can keep.
# 2. We will record main unit of measurement for this itemID as dose.

# In[7]:


valid_vupairs = [x for x in valid_vupairs if x[1] is not None]
valid_vupairs_des = sorted(valid_vupairs, key=lambda x: x[1])
for itemid, percentage, outputunits in valid_vupairs_des:
    print("IN "+str(itemid) + "\t" + "{:.2f}".format(percentage) + "\t" + str(len(outputunits))+" : "+ str(outputunits))
    
np.save(f'{res_dir}/filtered_input_raw.npy', {'raw': valid_vupairs})


# In[8]:


conn = getConnection()
sql = 'select hadm_id, amountuom, count(amountuom) from mimiciii.inputevents_cv where itemid={0} group by hadm_id, amountuom union all select hadm_id, amountuom, count(amountuom) from mimiciii.inputevents_mv where itemid={0} group by hadm_id, amountuom order by hadm_id'
for itemid in [x[0] for x in valid_vupairs_des[:14]]:
    cur = conn.cursor()
    cur.execute(sql.format(itemid))
    results = cur.fetchall()
    print('IN', itemid)
    print('hadm_id\t\tamountuom\tcount')
    for res in results:
        print('\t\t'.join(map(str, res)))
    print()


# In[9]:


valid_vupairs = np.load(f'{res_dir}/filtered_input_raw.npy', allow_pickle=True).tolist()['raw']
valid_input = [x[0] for x in valid_vupairs]
valid_input_unit = [x[2][0][0] for x in valid_vupairs]
print(valid_input, valid_input_unit)

np.save(f'{res_dir}/filtered_input.npy',{'id':valid_input,'unit':valid_input_unit})
print('saved!')


# In[10]:


print(len(admission_ids))

