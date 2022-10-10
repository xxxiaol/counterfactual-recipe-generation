# Source code and data for *Counterfactual Recipe Generation: Exploring Models’ Compositional Generalization Ability in a Realistic Scenario* (EMNLP2022 main conference paper)

---

## Dependencies
 - Python>=3.7
 - nltk
 - bert_score
 - ltp
 - sklearn

## Data
Our data is in the `data/` folder. 
Task data:
 - `dish_pairs.txt`: 50 pairs of (base dish, target dish)
 - `changing_ingres.txt`: 50 pairs of (original ingredient, changed ingredient)
 - `base_recipes.txt`: 2500 base recipes, each line is in the format of `base dish \t target dish \t base recipe`.
 - `recipe_corpus.txt`: 1,520,327 recipes of other dishes for model training. Link: https://drive.google.com/file/d/1gZqW_opPRA4ewTpetsIzOIJxYO-znKpu/view?usp=sharing
L2 evaluation data:
 - `glossary_dict.pkl`: glossary of ingredient classes, verb classes, and tool classes
 - `parsing_data.pkl`: data used in parsing recipes into actions
 - `pivot_actions.pkl`: pivot actions and order constraints

## Code
The code is in the `code/` folder.
Please change EVAL_TEXT_PATH to the path of generated recipes, and WORD_EMBEDDING_PATH to the word embedding path.
L1 evaluation (coverage of ingredients and extent of preservation): 
```
python L1_eval.py
```
L2 evaluation (action-level):
```
python L2_eval.py
```

## Citation
Please cite our paper if this repository inspires your work.
```
to be added
```

## Contact
If you have any questions regarding the code, please create an issue or contact the owner of this repository.