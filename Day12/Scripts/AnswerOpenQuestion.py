# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:48:40 2019

@author: isswan

16G extra harddisk 8G Mem
"""

##pip install wrapt --upgrade --ignore-installed
##pythoon -m deeppavlov install en_odqa_infer_wiki
##python -m deeppavlov install squad_bert
##

'''
Answer factoid questions from given Context
Pretrained BERT model trained  on SQuAD dataset
https://arxiv.org/abs/1810.04805
https://rajpurkar.github.io/SQuAD-explorer/
model config file:squad_bert.json
'''
from deeppavlov import build_model, configs

#model = build_model(configs.squad.squad, download=True)
model = build_model(configs.squad.squad, load_trained=True)

text = (" We can all think of at least one song that, when we hear it, "
        "triggers an emotional response. "
        "It might be a song that accompanied the first dance at your wedding,"
        " for example, or a song that reminds you of a difficult break-up or the loss of a loved one."
        " Given the deep connection we have with music,"
        " it is perhaps unsurprising that numerous studies have shown it can benefit our mental health."
        " A 2011 study by researchers from McGill University in Canada found that "
        "listening to music increases the amount of dopamine produced in the brain -"
        " a mood-enhancing chemical, making it a feasible treatment for depression.")

model([text], ['can music benifit health?'])
model([text], ['how can music benifit health?'])
model([text], ['why music can benifit health?'])
model([text], ['how music can be treatment for depression?'])



'''
Answerfactoid questions based on WIKIPedia
Core from on DrQA https://github.com/facebookresearch/DrQA/
https://arxiv.org/abs/1704.00051
Retriver: Bigram/TF-IDF weighted bag-of-word vectors. 
Reader: RNN model trained  on SQuAD dataset
model config file:en_odqa_infer_wiki
'''

from deeppavlov import configs
from deeppavlov.core.commands.infer import build_model

#odqa = build_model(configs.odqa.en_odqa_infer_wiki, download=True)
odqa = build_model(configs.odqa.en_odqa_infer_wiki, load_trained=True)

result = odqa(['where to go in Germany '])
print(result)


