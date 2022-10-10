import json
import gc
from tqdm import tqdm
import pickle as pkl
import numpy as np
from scipy import stats 
import re
from collections import Counter
import os

with open('../data/glossary_dict.pkl', 'rb') as f:
    ingre_dict, verb_dict, tools_dict=pkl.load(f)

from ltp import LTP
ltp = LTP(path="ltp_base.tgz")

def verb_match(w, verb_dict):
        i=len(w)
        while i>0 and w[:i] not in verb_dict:
            i-=1
        if i>0:
            return True, w[:i]
        else:
            return False, w
        

def text2actions(text):
    def recipe2phrases(ins, ltp):  # recipe -> phrases, segments, dependency list
        pattern_punc = r'[！？｡。，：；\.\!\?,:;]'
        pattern_num=r'0\.|,[0-9]*\.'
        step_list = re.split(pattern_num, ins)[1:]
        if len(step_list)<=1:
            step_list=[ins]
        dep_list=[]
        for k in step_list:
            cur_phrases=re.split(pattern_punc, k)
            for ph in cur_phrases:
                ph=ph.strip()
                if len(ph)==0 or '(' in ph or '（' in ph or ')' in ph or '）' in ph:
                    continue
                try:
                    seg, hidden = ltp.seg([ph])
                    dep = ltp.dep(hidden)
                    dep_list.append([ph, seg[0], dep[0]])
                except:
                    continue

        return dep_list

    def words_and_actions(dep_link, hd, cur):  # recursive: dep_link, cur head, cur node -> cur words, sub actions
        words=[cur]
        sub_actions=[]
        if cur in dep_link:  # not a leaf node
            for i in dep_link[cur]:
                if verb_match(i[0], verb_dict)[0] and i in dep_link:  # node i is a verb in the glossary
                    i_words, i_sub_actions=words_and_actions(dep_link, i, i)
                    sub_actions.append([i[0], i_words])
                    sub_actions.extend(i_sub_actions)
                else:
                    i_words, i_sub_actions=words_and_actions(dep_link, hd, i)
                    words.extend(i_words)
                    sub_actions.extend(i_sub_actions)

        return words, sub_actions

    def phrase2actions(ph, seg, dep):  # phrase -> action list
        dep_link={}
        for i in dep:
            x, y, z = i
            if y==0:
                hd=(seg[x-1], x-1)
            else:
                fr=seg[x-1]
                to=seg[y-1]
                if not (to, y-1) in dep_link:
                    dep_link[(to, y-1)]=[(fr, x-1)]
                else:
                    dep_link[(to, y-1)].append((fr, x-1))
        words_hd, sub_actions=words_and_actions(dep_link, hd, hd)
        if verb_match(hd[0], verb_dict)[0]:
            ph_a=[[hd[0], words_hd]]+sub_actions
        else:
            ph_a=sub_actions

        # sort by start position
        ph_start=[]
        for i in ph_a:
            ph_start.append(min([j[1] for j in i[1]]))
        ph_a_sorted=[x for _, x in sorted(zip(ph_start, ph_a))]

        # complete actions with no ingredient & tool using ingredients & tools of the previous action
        action_list=[]  # [verb, ingres, tools, original phrase, bool (ingredient is from previous or not)]
        action_end=[]
        for i in ph_a_sorted:
            cur_ingre=[j[0] for j in i[1] if j[0] in ingre_dict]
            cur_tool=[j[0] for j in i[1] if j[0] in tools_dict]
            cur_start=min([j[1] for j in i[1]])
            cur_end=max([j[1] for j in i[1]])
            action_end.append(cur_end)
            cur_ph="".join(seg[cur_start:cur_end+1])
            if len(cur_ingre)==0 and len(cur_tool)==0 and len(action_list)>0:
                cur_ingre=action_list[-1][1]
                action_list.append([i[0], frozenset(cur_ingre), frozenset(cur_tool), cur_ph, True])  # True: ingredient is from previous
            else:
                action_list.append([i[0], frozenset(cur_ingre), frozenset(cur_tool), cur_ph, False]) 

        # sort by end position
        action_list_sorted=[x for _, x in sorted(zip(action_end, action_list))]

        return action_list_sorted

    def phrases2actions(phs):
        actions=[]
        for i in phs:
            ph, seg, dep = i
            actions.append(phrase2actions(ph, seg, dep))

        # complete actions with no ingredient & tool using ingredients & tools of the previous action in the previous phrase
        action_list=[]
        prev_ingre = None
        for i in actions:
            if len(action_list)>0:
                prev_ingre=action_list[-1][1]
            for j in i:
                if len(j[1])==0 and len(j[2])==0 and prev_ingre is not None:
                    action_list.append([j[0], prev_ingre, j[2], j[3], True])
                else:
                    action_list.append(j)
        return action_list

    phs=recipe2phrases(text, ltp)
    actions=phrases2actions(phs)
    return actions


def replace_actions(actions, ingreA, ingreB):
    if ingreA=='':  # the task is to add ingreB
        return [], {}
    replaced=[]
    actions_o2r={}
    for i in actions:
        if ingreA in i[1]:
            tmp=set(i[1]).copy()
            tmp.remove(ingreA)
            tmp.add(ingreB)
            replaced.append([i[0], frozenset(tmp), i[2], i[3], i[4]])
            actions_o2r[tuple(i)]=len(replaced)-1
    return replaced, actions_o2r


def actions2proto(actions, dish_verb_dict, verb_action_dict):
    protos=[]
    for cur in actions:
        tmp=tuple([cur[1], cur[2]])
        _, v = verb_match(cur[0], verb_dict)
        if v in dish_verb_dict:
            v=dish_verb_dict[v]
            if tmp in verb_action_dict[v]:
                tmp=verb_action_dict[v][tmp]
        protos.append(tuple([v, tmp]))  # keep the original action if it is not in the proto actions
    return protos