#!/usr/bin/env python
# coding: utf-8

# # Filter Itemid Chart
# 
# This script is used for filtering itemids from TABLE CHARTEVENTS.
# 
# 1. We check number of units of each itemid and choose the major unit as the target of unit conversion.
# 2. In this step we get 3 kinds of features:
#     - numerical features
#     - categorical features
#     - ratio features, this usually happens in blood pressure measurement, such as "135/70".
# 
# ## Output
# 
# 1. itemid of observations for chartevents.
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
import re
from tqdm import tqdm_notebook as tqdm
from utils import getConnection

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


res_dir = "/media/data/mimic/res"


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


# In[3]:


def stat_chart_unit_task(ilist, admission_ids_txt):
    subresults = []
    tconn = getConnection()
    
    for i in tqdm(ilist):
        # for each itemID select number of rows group by unit of measurement.
        tcur = tconn.cursor()
        tcur.execute('SELECT coalesce(valueuom, \'\'), count(*) FROM mimiciii.chartevents WHERE itemid = '+ str(i) +' and hadm_id in (select * from admission_ids) group by valueuom')
        chartunits = tcur.fetchall()
        chartunits = sorted(chartunits, key=lambda tup: tup[1])
        chartunits.reverse()

        # count number of observation that has non numeric value
        tcur = tconn.cursor()
        tcur.execute('SELECT count(*) FROM mimiciii.chartevents WHERE itemid = '+ str(i) +' and hadm_id in (select * from admission_ids) and valuenum is null')
        notnum = tcur.fetchone()
        notnum = notnum[0]

        # total number of observation
        tcur = tconn.cursor()
        tcur.execute('SELECT count(*) FROM mimiciii.chartevents WHERE itemid = '+ str(i) +' and hadm_id in (select * from admission_ids)')
        total = tcur.fetchone()
        total = total[0]
        
        subresults.append((i, chartunits, notnum, total))
    
    tconn.close()
    return subresults

numworkers = cpu_count() // 2
# numworkers = 4
p = Pool(numworkers)
ilists = np.array_split(chart_itemid, numworkers)
results = [p.apply_async(stat_chart_unit_task, args=(ilist, admission_ids_txt)) for ilist in ilists]
p.close()
p.join()
results = [x.get() for x in results]
results = itertools.chain.from_iterable(results)
# results = []
# for i in tqdm(chart_itemid):
#     result = stat_chart_unit_task(i, admission_ids_txt)
#     results.append(result)
np.save(f'{res_dir}/filtered_chart_raw.npy', {'raw': results})
print('saved!')


# ## First filtering of categorical features
# 
# All features with numerical values < 80% of all records are possible categorical features. In this step we drop them for later analyzing.

# In[4]:


results = np.load(f'{res_dir}/filtered_chart_raw.npy', allow_pickle=True).tolist()['raw']
valid_chart = []
valid_chart_unit = []
valid_chart_cate = []
valid_chart_num = []
dropped = []
multiple_units = []
for x in results:
    i, chartunits, notnum, total = x[0], x[1], x[2], x[3]
    
    # calculate percentage of the top frequent unit compared to all observation.
    total2 = 0
    unitnum = 0
    for c in chartunits:
        total2 += c[1]
        if c[0] != '':
            unitnum += 1
    if total2 == 0:
        continue
    percentage = float(chartunits[0][1]) / total2 * 100.
    if unitnum > 1:
        multiple_units.append((i, chartunits, percentage))
    print("CHART "+str(i) + "\t" + "{:.2f}".format(percentage) +'\t'+ str(chartunits))
    
    # if the percentage of numeric number is less, then dropped it, and make it categorical feature.
    percentage =float(total -notnum)*100 / total
    print("Numeric observation :" + "{:.4f}%".format(percentage)+ " ( NOTNUM= " + str(notnum) + " / ALL= " + str(total) + " ) ")
    if(percentage < 80): 
        print('dropped\n')
        dropped.append(i)
        continue;
    print('')
    valid_chart.append(i)
    valid_chart_unit.append(chartunits[0][0])


# ## Unit inconsistency
# 
# Here are itemids having two or more different units.
# 
# For [211, 505], they have the same unit in fact. Keep them.
# 
# For [3451, 578, 113], the major unit covers > 90% of all records. Keep them.
# 
# For [3723], it is just a typo and we keep all.

# In[5]:


for i, chartunits, percentage in sorted(multiple_units, key=lambda x: x[2]):
    total2 = sum([t[1] for t in chartunits])
    percentage = float(chartunits[0][1]) / total2 * 100.
    print("CHART "+str(i) + "\t" + "{:.4f}".format(percentage) +'\t'+ str(chartunits))


# In[6]:


dropped_id = dropped
print(dropped_id, len(dropped_id))


# In[7]:


def numerical_ratio(units):
    res = list(map(lambda unit: re.match(r'(\d+\.\d*)|(\d*\.\d+)|(\d+)', unit), units))
    numerical_ratio = 1.0 * len([1 for r in res if r is not None]) / len(res)
    return numerical_ratio


# In[8]:


def dropped_value_list_unit_task(dropped_id):
    conn = getConnection()
    dropped_value = []
    for d in tqdm(dropped_id):
#         print('LAB : ' + str(d))
        cur = conn.cursor()
        cur.execute('SELECT value, valueuom, count(*) as x FROM mimiciii.chartevents as lb                     WHERE itemid = '+ str(d) +' and hadm_id in (select * from admission_ids) GROUP BY value, valueuom ORDER BY x DESC')
        droped_outs = cur.fetchall()
        if d == 4169:
            print(droped_outs)
        drop_array = []
        ct =0
        total = 0;
        for dx in droped_outs:
            total += dx[2]
#         print("Count ",total)
#         print([d[0] for d in droped_outs])
#         print('Numeric ratio', numerical_ratio([str(d[0]) for d in droped_outs]))
        units = []
        for dx in droped_outs:
            ct+=1
            if(ct>20):
                break
            dx = list(dx)
#             print(dx[1],dx[0],"\t",dx[2])
#         print('')
        dropped_value.append((d, droped_outs))
    conn.close()
    return dropped_value

dropped_value = []
numworkers = cpu_count() // 2
p = Pool(numworkers)
dropped_id_units = np.array_split(dropped_id, numworkers)
dropped_value_list = [p.apply_async(dropped_value_list_unit_task, args=(dropped_id_unit,)) for dropped_id_unit in dropped_id_units]
dropped_value_list = [x.get() for x in dropped_value_list]
dropped_value = list(itertools.chain.from_iterable(dropped_value_list))
np.save(f'{res_dir}/chart_dropped_value.npy', dropped_value)


# In[9]:


print(len(dropped_id))
print(len(valid_chart), len(valid_chart_unit))
print(valid_chart)
print(valid_chart_unit)


# ## Store selected features in first filtering
# 
# These features are all numerical features.

# In[10]:


np.save(f'{res_dir}/filtered_chart.npy',{'id':valid_chart,'unit':valid_chart_unit})
# np.save(f'{res_dir}/filtered_chart_cate',{'id':[223758],'unit':None})
print('saved!')


# ## Divide dropped features in first filtering
# 
# - Features with the ratio of non-numerical values(values that cannot pass the parser) > 0.5: categorical features
# - Features with the ratio of ratio values > 0.5: ratio features
# - otherwise: (possible) numerical features, we will parse them later

# In[11]:


dropped_value = np.load(f'{res_dir}/chart_dropped_value.npy', allow_pickle=True).tolist()
valid_chart_num = []
valid_chart_num_unit = []
valid_chart_cate = []
valid_chart_ratio = []
for d, droped_outs in dropped_value:
    ascnum = 0
    rationum = 0
    for value, valueuom, count in droped_outs:
        value = str(value)
        isasc = re.search(r'(\d+\.\d*)|(\d*\.\d+)|(\d+)', value) is None
        isratio = re.fullmatch(r'{0}\/{0}'.format(r'((\d+\.\d*)|(\d*\.\d+)|(\d+))'), value) is not None
        if isasc:
            ascnum += 1
        if isratio:
            rationum += 1
    if d == 4169:
        print(ascnum, len(droped_outs), droped_outs, rationum)
    if ascnum / len(droped_outs) >= 0.5:
        valid_chart_cate.append(d)
    elif rationum / len(droped_outs) >= 0.5:
        valid_chart_ratio.append(d)
        # print(droped_outs)
    else:
        valid_chart_num.append(d)
        if droped_outs[0][1] is None:
            valid_chart_num_unit.append('')
        else:
            valid_chart_num_unit.append(droped_outs[0][1])
#         print(droped_outs)
        
print(len(valid_chart_num), len(valid_chart_cate), len(valid_chart_ratio))
print(valid_chart_num, valid_chart_num_unit, valid_chart_ratio)


# ## Store 3 kinds of features

# In[12]:


print(len(valid_chart_num), len(valid_chart_num_unit), len(valid_chart_cate))
print(valid_chart_num, valid_chart_num_unit, valid_chart_cate)
np.save(f'{res_dir}/filtered_chart_num',{'id':valid_chart_num,'unit':valid_chart_num_unit})
np.save(f'{res_dir}/filtered_chart_cate',{'id':valid_chart_cate,'unit':None})
np.save(f'{res_dir}/filtered_chart_ratio', {'id': valid_chart_ratio, 'unit': None})

