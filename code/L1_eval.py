import numpy as np
import os
from bert_score import score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

dish_pairs=[]
with open('../data/dish_pairs.txt', 'r') as f:
    for line in f:
        splits=line.strip().split(' ')
        dish_pairs.append(splits)

ingres_r=[]  # replaced ingredient
ingres_a=[]  # added ingredient
with open('../data/changing_ingres.txt', 'r') as f:
    for line in f:
        splits=line.split()
        if len(splits)==1:
            splits=['']+splits
        ingres_r.append(splits[0])
        ingres_a.append(splits[1])
        
original_text=[]
with open('../data/base_recipes.txt', 'r') as f:
    for line in f:
        splits=line.split('\t')
        original_text.append(splits[2].strip())
        
eval_text=[]
with open(EVAL_TEXT_PATH, 'r') as f:
    for line in f:
        eval_text.append(line.strip())
        
def ingre_coverage(ingres, text):
    ingres_dupl=[]
    num=50
    for i in ingres:
        for j in range(num):
            ingres_dupl.append(i)
    
    covered=0
    all_cnt=0
    for i in range(len(ingres_dupl)):
        if ingres_dupl[i]!='':
            all_cnt+=1
            if ingres_dupl[i] in text[i]:
                covered+=1
            
    return covered/all_cnt


from ltp import LTP
ltp = LTP(path="ltp_base.tgz")

def split(text):
    splited=[]
    for i in text:
        if isinstance(i, str):
            seg, _ = ltp.seg([i])
            splited.append(seg[0])
        else:
            cur=[]
            for j in i:
                seg, _ = ltp.seg([j])
                cur.append(seg[0])
            splited.append(cur)
    return splited


def calc_bert_score(original_text, eval_text):
    cands=eval_text
    cands=[i[:510] for i in cands]  # length limit: 512
    refs=[i[:510] for i in original_text]
    scores=score(cands, refs, lang='zh', verbose=True)
    return [np.mean(scores[i].detach().cpu().numpy()) for i in range(3)]


def calc_scores(ingres_r, ingres_a, original_text, eval_text):
    ingre_cover_added=ingre_coverage(ingres_a, eval_text)
    print('Ingredient coverage (Added):', ingre_cover_added)
    ingre_cover_replaced=ingre_coverage(ingres_r, eval_text)
    print('Ingredient coverage (Replaced):', ingre_cover_replaced)
    
    original_text_split=split(original_text)
    original_text_split=[[i] for i in original_text_split]
    eval_text_split=split(eval_text)
    
    # The default BLEU calculates a score for up to 4-grams using uniform weights
    bleu_o = corpus_bleu(original_text_split, eval_text_split)
    bertscore_o =calc_bert_score(original_text, eval_text)[2]
    print('Extent of preservation | bleu-4:', bleu_o, 'bertscore:', bertscore_o)
    
    return ingre_cover, bleu_o, bertscore_o


calc_scores(ingres_r, ingres_a, original_text, eval_text)