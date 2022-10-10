import pickle as pkl
import re
import numpy as np
import os
from collections import Counter
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from parse_functions import *

dish_pairs=[]
with open('../data/dish_pairs.txt', 'r') as f:
    for line in f:
        splits=line.strip().split(' ')
        dish_pairs.append(splits)

ingres=[]
with open('../data/changing_ingres.txt', 'r') as f:
    for line in f:
        splits=line.strip().split(' ')
        if len(splits)==1:
            splits=['']+splits
        ingres.append(splits)
        
original_text=[]
with open('../data/base_recipes.txt', 'r') as f:
    for line in f:
        splits=line.split('\t')
        original_text.append(splits[2].strip())
        
eval_text=[]
with open(EVAL_TEXT_PATH, 'r') as f:
    for line in f:
        eval_text.append(line.strip())
        
with open('../data/parsing_data.pkl', 'rb') as f:
    parsing_data=pkl.load(f)
    
with open('../data/pivot_actions.pkl', 'rb') as f:
    all_pivot_actions, all_constraints=pkl.load(f)
    
wv = WORD_EMBEDDING_PATH  #  we use Tencent AI Lab embeddings (https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d200-v0.2.0.tar.gz), and other word embeddings can also be used
        
        
def find_reference_text_for_action(references_actions, action, dish_verb_dict, verb_action_dict):
    action_text=[]
    for i in references_actions:
        protos=actions2proto(i, dish_verb_dict, verb_action_dict)
        for j in range(len(protos)):
            if protos[j]==action:
                action_text.append(i[j][3])
    return action_text


def validate_orders(correct_insertion, proto_g, constraints):
    wrong_order_actions=[]
    for i in constraints:  # (action_to_insert, list of (cause, effect))
        if i[0] in correct_insertion:
            cur_num=0
            cur_correct=0
            for j in i[1]:
                pos_c=-1
                pos_e=-1
                for idxk, k in enumerate(proto_g):
                    if j[0]==k:
                        pos_c=idxk
                    if j[1]==k:
                        pos_e=idxk
                if pos_c>=0 and pos_e>=0:
                    cur_num+=1
                    if pos_c<pos_e:
                        cur_correct+=1

            if cur_correct<cur_num:
                wrong_order_actions.append(i[0])
                
    return wrong_order_actions


text2embedding={}
def text_embeddings(text):
    embeddings=[]
    for i in text:
        if not i in text2embedding:
            cur=np.zeros(200)
            try:
                seg, hidden = ltp.seg([i])
                cnt=0
                for j in seg[0]:
                    if j in wv:
                        cur+=wv[j]
                        cnt+=1
            except Exception as e:
                print(i, e)
            if cnt>0:
                text2embedding[i]=cur/cnt
            else:
                text2embedding[i]=cur
        embeddings.append(text2embedding[i])
    return np.array(embeddings)


def calc_text_sim(text1, text2):
    embeddings1=text_embeddings(text1)
    embeddings2=text_embeddings(text2)
    return cosine_similarity(embeddings1, embeddings2)


def calc_f1(p, r):
    if p==0 and r==0:
        return 0
    else:
        return np.divide(2*p*r, p+r)

    
def measure_metrics(set1, set2, wrong_order_actions):  
    # set1: pivot_text (dict of (proto: list of strings)), set2: generate_text (dict of (proto: string))
    if len(set1)==0 and len(set2)==0:
        return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
    if len(set1)==0:
        return 0, np.NaN, np.NaN, 0, np.NaN, np.NaN
    if len(set2)==0:
        return np.NaN, 0, np.NaN, np.NaN, 0, np.NaN
    
    hard_precision=0
    hard_recall=0
    soft_precision=0
    soft_recall=0
    
    threshold=0.9
    candidate_texts=[set2[j] for j in set2 if not j in wrong_order_actions]
    candidate_actions=[j for j in set2 if not j in wrong_order_actions]
   
    if len(candidate_texts)>0:
        biadjacency_matrix=np.zeros((len(set1), len(candidate_actions)))
        for idxi, i in enumerate(set1):
            if len(set1[i])>0:
                sim_scores=calc_text_sim(set1[i], candidate_texts)
                sim_scores[sim_scores<threshold]=0
                biadjacency_matrix[idxi]=np.max(sim_scores, axis=0)
            for idxj in range(len(candidate_actions)):
                if i == candidate_actions[idxj]:
                    biadjacency_matrix[idxi][idxj]=1

        for i in range(len(set1)):
            if np.max(biadjacency_matrix[i])>1-1e-5:
                hard_recall+=1
            soft_recall+=np.max(biadjacency_matrix[i])
                
        for i in range(len(candidate_texts)):
            if np.max(biadjacency_matrix[:, i])>1-1e-5:
                hard_precision+=1
            soft_precision+=np.max(biadjacency_matrix[:, i])
    
    hard_precision/=len(set2)
    hard_recall/=len(set1)
    hard_f1=calc_f1(hard_precision, hard_recall)
    soft_precision/=len(set2)
    soft_recall/=len(set1)
    soft_f1=calc_f1(soft_precision, soft_recall)
    
    return hard_precision, hard_recall, hard_f1, soft_precision, soft_recall, soft_f1


def calc_one(ingreA, ingreB, o_text, g_text, pivot_actions, 
             dish_verb_dict, verb_action_dict, constraints, 
             base_references_actions, target_references_actions):
    # parse text to proto-actions
    actions_g=text2actions(g_text)
    proto_g=actions2proto(actions_g, dish_verb_dict, verb_action_dict)
    actions_o=text2actions(o_text)
    proto_o=actions2proto(actions_o, dish_verb_dict, verb_action_dict)
    
    # find actions that simply replace ingredients (they are not counted)
    actions_r, actions_o2r=replace_actions(actions_o, ingreA, ingreB)
    proto_r=actions2proto(actions_r, dish_verb_dict, verb_action_dict)
    proto_o2r={proto_o[i]: proto_r[actions_o2r[tuple(actions_o[i])]] 
                   for i in range(len(proto_o)) if tuple(actions_o[i]) in actions_o2r}
    pivot_sets=[set(), set()]  # pivot actions to remove/insert that meet the current base recipe
    pivot_text=[{}, {}]  # text of the pivot actions (for calculating text similarity in the soft metric)
    for s in pivot_actions:
        cur_action=None
        # s[0]: [verb, action]
        # s[1]: 0 (remove) / 1 (insert)
        if s[1]==0 and tuple([s[0][0], s[0][1]]) in proto_o: # the action is in the base recipe
            cur_action=tuple([s[0][0], s[0][1]])
            cur_idx=0
        elif s[1]==1 and tuple([s[0][0], s[0][1]]) not in proto_o:
            cur_action=tuple([s[0][0], s[0][1]])
            cur_idx=1
        if cur_action is not None:
            pivot_sets[cur_idx].add(cur_action)
            if cur_idx==1:  # insert (find reference text from recipes of target dish)
                cur_text=find_reference_text_for_action(target_references_actions, 
                                                        cur_action, dish_verb_dict, verb_action_dict)
            else:  # remove (find reference text from recipes of base dish)
                cur_text=find_reference_text_for_action(base_references_actions, 
                                                        cur_action, dish_verb_dict, verb_action_dict)
            pivot_text[cur_idx][cur_action]=cur_text
    
    generate_sets=[set(), set()]
    generate_text=[{}, {}]
            
    for idxi, i in enumerate(proto_o):
        if not i in proto_g:  # the action is removed in the generated text
            if i in pivot_sets[0] or not (i in proto_o2r and proto_o2r[i] in proto_g):  # not simple replacement
                generate_sets[0].add(i)
                generate_text[0][i]=actions_o[idxi][3]

    for idxi, i in enumerate(proto_g):
        if not i in proto_o:  # the action is added in the generated text
            if i in pivot_sets[1] or not i in proto_r:  # not simple replacement
                generate_sets[1].add(i)
                generate_text[1][i]=actions_g[idxi][3]
                
    correct_insertion=pivot_sets[1].intersection(generate_sets[1])
    wrong_order_actions = validate_orders(correct_insertion, proto_g, constraints)

    pivot_all=pivot_text[0].copy()
    pivot_all.update(pivot_text[1])
    generate_all=generate_text[0].copy()
    generate_all.update(generate_text[1])
    score=measure_metrics(pivot_all, generate_all, wrong_order_actions)
    
    return score


def calc(dish_pairs, original_text, eval_text):
    scores=[]
    num=50  # 50 recipes for one dish pair
    for i in tqdm(range(len(dish_pairs))):
        dishA, dishB = dish_pairs[i]
        ingreA, ingreB = ingres[i]
        dish_verb_dict, verb_action_dict, base_references_actions, target_references_actions = parsing_data[dishB]
        pivot_actions = all_pivot_actions[dishB]
        constraints = all_constraints[dishB]
        for j in range(num):
            score=calc_one(ingreA, ingreB, original_text[i*num+j], eval_text[i*num+j], 
                           pivot_actions, dish_verb_dict, verb_action_dict, 
                           constraints, base_references_actions, target_references_actions)
            scores.append(score)
        
    return np.nanmean(scores, axis=0) 


calc(dish_pairs, original_text, eval_text)