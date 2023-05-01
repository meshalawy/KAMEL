#%%
import ray 
import logging 
import os

import sys
sys.path.insert(0, os.getcwd())

from kamel import KAMEL

from itertools import repeat, chain
from shapely.geometry import LineString
from utils import *
import pickle

#%%

@ray.remote(num_cpus=2, max_retries=-1, scheduling_strategy="SPREAD")
def impute_a_gap(input_points, gap_at, models):
    self = ray.get(models['model'])
    return self.impute_a_gap(input_points, gap_at)



"""
Takes list of points to perform imputations between every consecutive pair 
The input points must be of 4326 projection, longitude and latitiude.
each pair or points is called segment.
returns a dict with the following six keys: 

#  imputed_noseg_pt : imputed trajectory no segments represented as list of points
#  imputed_seg_pt : imputed trajectory segmentes represented as list of points
#  inferred_noseg_pt : the additional inferred points only with no segments represented as list of points
#  inferred_seg_pt : the additional inferred points only with segments represented as list of points
#  imputed_noseg_ls : imputed trajectory no segments represented as a line string
#  imputed_seg_ls : imputed trajectory segmentes represented as a line string

"""

@ray.remote(num_cpus=1, max_retries=-1, scheduling_strategy="SPREAD")
def get_imputed_trajectory(input_points, models):

    imputed_noseg_pt = []
    imputed_seg_pt = []
    inferred_noseg_pt = []
    inferred_seg_pt = []
    imputed_noseg_ls = None
    imputed_seg_ls = None
    

    if len(input_points)>=2: 

        gaps_at = range(len(input_points)-1)
        imputations_for_all_segments = []
        
        refs = [impute_a_gap.remote(*args) for args in zip(repeat(input_points), gaps_at, repeat(models))]
        imputations_for_all_segments = ray.get(refs)
        
        imputed_seg_pt = [r['imputed_seg_pt'] for r in imputations_for_all_segments]
        inferred_seg_pt = [r['inferred_seg_pt'] for r in imputations_for_all_segments]
        imputed_seg_score = [r['imputed_seg_score'] for r in imputations_for_all_segments]
        imputed_seg_time = [r['imputed_seg_time'] for r in imputations_for_all_segments]
        imputed_noseg_time = sum(imputed_seg_time)

        # imputed_noseg_pt:
        for segment in imputed_seg_pt:
            imputed_noseg_pt = imputed_noseg_pt + segment[:-1]

        imputed_noseg_pt.append(segment[-1])

    
    # inferred_noseg_pt
    inferred_noseg_pt = list(chain(*inferred_seg_pt))
    
    # representation as linestrings: takes the points and create a linestring from them
    if len(imputed_noseg_pt) >= 2: 
        imputed_noseg_ls = LineString(imputed_noseg_pt)

    imputed_seg_ls = list(map(LineString, imputed_seg_pt))

    return {
        'imputed_noseg_pt' : imputed_noseg_pt,
        'imputed_seg_pt' : imputed_seg_pt,
        'inferred_noseg_pt' : inferred_noseg_pt,
        'inferred_seg_pt' : inferred_seg_pt,
        'imputed_noseg_ls' : imputed_noseg_ls,
        'imputed_seg_ls' : imputed_seg_ls,
        'imputed_seg_score' : imputed_seg_score,
        'imputed_seg_time' : imputed_seg_time,
        'imputed_noseg_time': imputed_noseg_time
    }
    
    
#%%
    

args = {
    "imputer": 'BERT',
    "imputer_args":{
        "bert_dir": 'models/porto',
        "detokenizer": "token2point_cluster_centroid",
        "beam_size": 10,
        "beam_normalization": 0.7,
        "use_constraints": True
    }
}


imputer = BERTImputer(**args['imputer_args'])
imputer.init_models()

ray.init()
imputer = ray.put(imputer)



# Load data from its source. For example Porto dataset:
# https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data

# points = [[...]]

imputation_results = execute_ray_parallel_stateless3(
    get_imputed_trajectory,
    [[i] for i in points[:]],
    kw_args={
        "models":{
            "model":imputer
        }
    }
)


with open(f'out.pkl', 'wb') as f:
    pickle.dump(imputation_results,f)
    
