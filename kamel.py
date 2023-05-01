

# %%
from math import log10
import h3
from shapely.geometry import LineString, Point
from transformers import pipeline 
from types import SimpleNamespace
import ray
import pickle
import itertools
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from itertools import repeat
import time
from bearing import calculate_bearing
import numpy as np
#%%


class KAMEL(object):
    
    
    bert_dir= None
    bert_model = None
    h3_clusters = None
    h3_resolution = None
    
    detokenizer = None
    beam_size = None
    use_constraints = None

    models_initialized = False
    
    def __init__(self, bert_dir, detokenizer, 
            beam_size, beam_normalization, 
            use_constraints):

        self.bert_dir = bert_dir
        self.detokenizer = getattr(self, detokenizer)
        self.beam_size = beam_size
        self.beam_normalization = beam_normalization
        self.use_constraints = use_constraints


    def init_models(self):
        if not self.models_initialized:
            from transformers import logging
            logging.set_verbosity_error()
            self.bert_model = pipeline('fill-mask', model=self.bert_dir, top_k=self.beam_size)

            if self.detokenizer == self.token2point_cluster_centroid:
                with open(f'{self.bert_dir}/clusters.pkl', 'rb') as file:
                    self.h3_kmeans = pickle.load (file)

            with open(f'{self.bert_dir}/resolution.txt', 'r') as file: 
                self.h3_resolution = int(file.readlines()[0])

            self.models_initialized = True
    

    def log(self,message):
        pass

    def points2tokens (self, points):
        return [h3.geo_to_h3(p.y,p.x, self.h3_resolution) 
                for p in points]

    

    def token2point_h3_centroid(self, token, previous_point=None):
        # returns centroid of the hexagon
        y, x = h3.h3_to_geo(token)
        return Point(x,y)

    def token2point_data_centroid(self, token, previous_point=None):
        # return centroid of "all data" in the hexagon
        if token in self.h3_clusters:
            cluster = self.h3_clusters[token]
            x, y = cluster['x'], cluster['y']
            return Point(x,y)
        else:
            return self.token2point_h3_centroid(token, None)

    
    def token2point_cluster_centroid(self, token, previous_point):
        # return centroid of the closest cluster 
        c = self.token2point_data_centroid(token, None)
        
        if token not in self.h3_kmeans:
            return c
        
        if token in self.h3_clusters and self.h3_clusters[token]['current_count'] <= 20:
            return c

        angle = calculate_bearing((previous_point.y, previous_point.x), (c.y, c.x))
        
        m, means = self.h3_kmeans[token]
        x, y, _ = means[m.predict(np.array([angle]).reshape(-1,1))][0]
        return Point(x,y)


    def impute_a_gap(self, input_points, gap_at):

        start = time.time()

        imputed_seg_pt = input_points[gap_at : gap_at + 2]
        inferred_seg_pt = []

        h3_input = self.points2tokens(input_points)    
        part1 = h3_input[0:gap_at + 1]
        part2 = h3_input[gap_at + 1:]

        p_from = part1[-1] 
        p_to = part2[0]

        max_length =  2 * h3.h3_distance(p_from, p_to)

        self.log(f"""imputing at gap {gap_at}
        part 1 : {part1}
        part 2 : {part2}
        
        """)

        
        most_likely_sequence, score = self.beam_search(part1, part2, other_segments_imputations=[], max_length=max_length)
        # most_likely_sequence, score = self.beam_search(part1, part2, other_segments_imputations=[])
        
        for h3_token in most_likely_sequence:
            p = self.detokenizer(h3_token, imputed_seg_pt[-2])
            
            # insert the point at imputed_seg_pt[-1], i.e. before the last one
            # because we are imputing between two points. 

            imputed_seg_pt.insert(-1, p)
            inferred_seg_pt.append(p)

        end = time.time()
        imputed_seg_time = (end - start)

        self.log(f'done at gap {gap_at} with score {score}')
        return {
            'imputed_seg_pt': imputed_seg_pt,
            'inferred_seg_pt': inferred_seg_pt,
            'imputed_seg_score': score,
            'imputed_seg_time': imputed_seg_time
        }

    def call_bert(self, part1, part2):

        input = part1 +  ['[MASK]'] + part2
        input = ' '.join(input)
        
        
        json = self.bert_model(input)
        
        # TODO: check if API successed and json is not empty
        results = {
            'ok': True,
            'json': json
        }
        return SimpleNamespace(**results)

    def get_constrained_candidates_between_two_points(self, p_from, p_to, factor=1):
        dist = h3.h3_distance(p_from,p_to)
        dist = round(dist * factor)
        ring1 = h3.k_ring(p_from, dist)
        ring2 = h3.k_ring(p_to, dist)
        constrained_candidates = set(ring1).intersection(ring2)
        return set(constrained_candidates)

    def beam_search(self, part1, part2, 
                other_segments_imputations = [],
                max_length = None,
                beam_size = None,
                beam_normalization = None):


        
        beam_size = beam_size or self.beam_size
        beam_normalization = beam_normalization or self.beam_normalization   

        self.log(f'beam search is called for the follwoing points: {[part1[-1], part2[0], other_segments_imputations]}')

        # initial empty beam results to build on
        beam_results = [{
            "sequence": [],
            "score": 0,
            "normalized_score": 0,
            "has_no_more_candidates": False
        }]

        org_constined_cand = set(
            self.get_constrained_candidates_between_two_points(
                                        part1[-1], part2[0]))



        for i in range(max_length):
            new_solutions = []
            for result in beam_results:
                if result["has_no_more_candidates"]:
                    # keep the results in our solution list, but no need to try to check for 
                    # new following points since we didn't find any thing in the last step i-1
                    # so we will get the same results if we call again.
                    new_solutions.append(result)
                    continue

                sequence = result["sequence"]
                found_at_least_1_candidate = False


                for gap_placement in range(len(sequence)+1):
                    modified_part1 =  part1 + sequence[:gap_placement]
                    modified_part2 = sequence[gap_placement:] + part2

                    p_from = modified_part1[-1] 
                    p_to = modified_part2[0]

                    if h3.h3_distance(p_from, p_to) <=1:
                        continue

                    # prepare the constrained candidates
                    constrained_cand = org_constined_cand.intersection(
                        self.get_constrained_candidates_between_two_points(p_from, p_to))
                
                    self.log(f"""calling bert as folows: 
                    part 1 {modified_part1}
                    part 2 {modified_part2}
                    """)
                    res = self.call_bert(modified_part1, modified_part2)

                    predictions = res.json
                    

                    # bert will return the top x (x=beam_size because it is intialized to do so in init_models)
                    for p in predictions:
                        candidate = p['token_str']
                        score =  p['score']
                        self.log(f'new candidate {candidate} with score {score}')

                        if candidate == p_from or candidate == p_to:
                            self.log('skipped because it matches the gap boundaries')
                            continue

                        if candidate in sequence:
                            self.log('skipped because it is already in the current sequence')
                            continue
                        

                        if self.use_constraints and candidate not in constrained_cand:
                            self.log('skipped because it is not allowed/ not among the constrained candidates')
                            continue

                        
                        found_at_least_1_candidate = True
                        

                        new_sequence = [*sequence]
                        new_sequence.insert(gap_placement, candidate)

                        new_score = result["score"] + (-1 * log10(score))
                        
                        # normalize by the length
                        new_normalized_score = new_score / (len(new_sequence)**beam_normalization)

                        new_solutions.append({
                            "sequence": new_sequence,
                            "score": new_score,
                            "normalized_score": new_normalized_score,
                            "has_no_more_candidates": False
                        })
                        self.log (f'new solution appended as follow. \n {new_solutions[-1]}')


                # if no more candidate at this step i, then there will be no candidates as weel at i+1
                # so no need to check again at the next time step. we flag has_no_more_candidate as True
                # so the next iteration can see that and skit it. But we need to keep the result so 
                # it doesn't get lost
                if not found_at_least_1_candidate:
                    # No more candidates. Append the current results with updated has_no_more_candidates
                    result["has_no_more_candidates"] = True
                    new_solutions.append(result)
                    self.log (f'No more Candidates for the sequence {sequence}')

            
            self.log(f"beam results at {i} was: {beam_results}")
            
            # beam_results now may include duplicates. we will sort then iterate and keep only distinct solutions 
            
            beam_results = sorted(new_solutions, key=lambda s: s['normalized_score'])
            distinct_results = []
            seen_sequences = []
            for solution in beam_results:
                sequence = set(solution['sequence'])
                if sequence in seen_sequences:
                    continue
                else:
                    distinct_results.append(solution)
                    seen_sequences.append(sequence)
            

            beam_results = distinct_results[:beam_size]
            self.log(f'current beam results at end of step i {i} is : \n {beam_results}')
        
        
        top_result = beam_results[0]
        top_sequence = top_result['sequence']
        top_score = top_result['normalized_score']

        self.log(f'beam finished with score: {top_score}' )
        return top_sequence, top_score

        
# %%
