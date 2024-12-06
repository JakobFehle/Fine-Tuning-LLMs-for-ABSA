import pandas as pd
import re
import json
import torch
import os

from itertools import product
from ast import literal_eval

DATASETS = ['GERestaurant', 'rest-16']

LABEL_SPACES = [      
    ###
    #  GERestaurant
    ###
['AMBIENCE:NEGATIVE','AMBIENCE:NEUTRAL','AMBIENCE:POSITIVE','FOOD:NEGATIVE','FOOD:NEUTRAL','FOOD:POSITIVE','PRICE:NEGATIVE','PRICE:NEUTRAL','PRICE:POSITIVE','GENERAL-IMPRESSION:NEGATIVE','GENERAL-IMPRESSION:NEUTRAL','GENERAL-IMPRESSION:POSITIVE','SERVICE:NEGATIVE','SERVICE:NEUTRAL','SERVICE:POSITIVE'],
    ###
    #  SemEval Rest-16
    ###
    ['AMBIENCE#GENERAL:POSITIVE', 'AMBIENCE#GENERAL:NEUTRAL', 'AMBIENCE#GENERAL:NEGATIVE', 'DRINKS#PRICES:POSITIVE', 'DRINKS#PRICES:NEUTRAL', 'DRINKS#PRICES:NEGATIVE', 'DRINKS#QUALITY:POSITIVE', 'DRINKS#QUALITY:NEUTRAL', 'DRINKS#QUALITY:NEGATIVE', 'DRINKS#STYLE_OPTIONS:POSITIVE', 'DRINKS#STYLE_OPTIONS:NEUTRAL', 'DRINKS#STYLE_OPTIONS:NEGATIVE', 'FOOD#PRICES:POSITIVE', 'FOOD#PRICES:NEUTRAL', 'FOOD#PRICES:NEGATIVE', 'FOOD#QUALITY:POSITIVE', 'FOOD#QUALITY:NEUTRAL', 'FOOD#QUALITY:NEGATIVE', 'FOOD#STYLE_OPTIONS:POSITIVE', 'FOOD#STYLE_OPTIONS:NEUTRAL', 'FOOD#STYLE_OPTIONS:NEGATIVE', 'LOCATION#GENERAL:POSITIVE', 'LOCATION#GENERAL:NEUTRAL', 'LOCATION#GENERAL:NEGATIVE', 'RESTAURANT#GENERAL:POSITIVE', 'RESTAURANT#GENERAL:NEUTRAL', 'RESTAURANT#GENERAL:NEGATIVE', 'RESTAURANT#MISCELLANEOUS:POSITIVE', 'RESTAURANT#MISCELLANEOUS:NEUTRAL', 'RESTAURANT#MISCELLANEOUS:NEGATIVE', 'RESTAURANT#PRICES:POSITIVE', 'RESTAURANT#PRICES:NEUTRAL', 'RESTAURANT#PRICES:NEGATIVE', 'SERVICE#GENERAL:POSITIVE', 'SERVICE#GENERAL:NEUTRAL', 'SERVICE#GENERAL:NEGATIVE']]

TRANSLATE_POLARITIES_EN = {'POSITIVE': 'positively', 'NEUTRAL': 'neutrally', 'NEGATIVE': 'negatively'}
TRANSLATE_POLARITIES_GER = {'POSITIVE': 'positiv', 'NEUTRAL': 'neutral', 'NEGATIVE': 'negativ'}

REGEX_ASPECTS_ACSD = r"\(([^,]+),[^,]+,\s*\"[^\"]*\"\)"
REGEX_LABELS_ACSD = r"\([^,]+,\s*([^,]+)\s*,\s*\"[^\"]*\"\s*\)"
REGEX_PHRASES_ACSD = r"\([^,]+,\s*[^,]+\s*,\s*\"([^\"]*)\"\s*\)"
REGEX_ASPECTS = r"\(([^,]+),[^)]+\)"
REGEX_LABELS = r"\([^,]+,\s*([^)]+)\)"

def raise_err(ex):
    raise ex

def loadDataset(data_path, dataset_name, low_resource_setting, task, split = 0,  original_split = False):
    if dataset_name not in DATASETS:
        print('Dataset name not valid!')
        return None, None, None

    DATASET_PATH = data_path if task != 'e2e-e' else data_path + '/data_e2e_e'
    
    label_space = LABEL_SPACES[DATASETS.index(dataset_name)]

    fn_suffix = '_full' if low_resource_setting == 0 else f'_{low_resource_setting}'
    
    if original_split:
        path_train = f'{DATASET_PATH}/{dataset_name}/train.tsv'
        path_eval = f'{DATASET_PATH}/{dataset_name}/test.tsv'

    else:
        if split != 0: # CV Test Phase
            path_train = f'{DATASET_PATH}/{dataset_name}/split_{split}/train{fn_suffix}.tsv'
            path_eval = f'{DATASET_PATH}/{dataset_name}/split_{split}/test_full.tsv'
            
        else: # HT Phase
            path_train = f'{DATASET_PATH}/{dataset_name}/train{fn_suffix}.tsv'
            path_eval = f'{DATASET_PATH}/{dataset_name}/val{fn_suffix}.tsv'

    converters = {'labels': literal_eval, 'labels_phrases': literal_eval}
    
    df_train = pd.read_csv(path_train, sep='\t', converters=converters).set_index('id')
    df_eval = pd.read_csv(path_eval, sep='\t', converters=converters).set_index('id')
    
    print(f'Loading dataset ...')
    print(f'Dataset name: {dataset_name}')
    print(f'Split setting: ', 'Original' if original_split else 'Custom')
    print(f'Eval Mode: ', 'Test' if split != 0 else 'Validation')
    print(f'Low Resource Setting: ', 'Full' if low_resource_setting == '0' else low_resource_setting)
    print(f'Train Length: ', len(df_train))
    print(f'Eval Length: ', len(df_eval))
    print(f'Split: {split}')
        
    return df_train, df_eval, label_space
    
def createCoTText(few_shot_template, absa_task, examples_text, examples_labels, lang):
    if absa_task == 'acsa':
        
        re_aspects = examples_labels[0]
        re_labels = examples_labels[1]

        if len(re_aspects) == 1:
            if lang == 'ger':
                aspect_labels = 'Zuerst identifizieren wir hierfür das im Satz angesprochene Aspekt: ' + re_aspects[0]
                sentiments = 'Anschließend ermitteln wir das Sentiment, welches gegenüber diesem Aspekt ausgedrückt wird: ' + 'Das Aspekt ' + re_aspects[0] + ' wird im Text ' + TRANSLATE_POLARITIES_GER[re_labels[0]] + ' bewertet'
                labels_text = 'Das finale Ergebnis besteht somit aus dem folgenden Aspekt-Sentiment-Paar: '
            elif lang == 'en':
                aspect_labels = 'First, we identify the aspect addressed in the sentence: ' + re_aspects[0]
                sentiments = 'Next, we determine the sentiment expressed towards this aspect: ' + 'The aspect ' + re_aspects[0] + ' is referenced ' + TRANSLATE_POLARITIES_EN[re_labels[0]] + ' in the text'
                labels_text = 'The final result thus consists of the following aspect-sentiment-pair: '
        elif len(re_aspects) == 2: 
            if lang == 'ger':
                aspect_labels = 'Zuerst identifizieren wir hierfür die im Satz angesprochenen Aspekte: ' + re_aspects[0] + ' und ' + re_aspects[1]
                sentiments = 'Anschließend ermitteln wir das Sentiment, welches gegenüber diesen Aspekten ausgedrückt wird: ' + 'Das Aspekt ' + re_aspects[0] + ' wird im Text ' + TRANSLATE_POLARITIES_GER[re_labels[0]] + ' und das Aspekt ' + re_aspects[1] + ' wird im Text ' + TRANSLATE_POLARITIES_GER[re_labels[1]] + ' bewertet'
                labels_text = 'Das finale Ergebnis besteht somit aus den folgenden Aspekt-Sentiment-Paaren: '
            elif lang == 'en':
                aspect_labels = 'First, we identify all aspects addressed in the sentence: ' + re_aspects[0] + ' and ' + re_aspects[1]
                sentiments = 'Next, we determine the sentiment expressed towards these aspects: ' + 'The aspect ' + re_aspects[0] + ' is referenced ' + TRANSLATE_POLARITIES_EN[re_labels[0]] + ' and the aspect ' + re_aspects[1] + ' is referenced ' + TRANSLATE_POLARITIES_EN[re_labels[1]] + ' in the text'
                labels_text = 'The final result thus consists of the following aspect-sentiment-pairs: '
        elif len(re_aspects) > 2: 
            if lang == 'ger':
                aspect_labels = 'Zuerst identifizieren wir hierfür die im Satz angesprochenen Aspekte: ' + ', '.join(re_aspects[:-1]) + ' und ' + re_aspects[-1]
                       
                sentiment_expressions = []
            
                for j, asp in enumerate(re_aspects):
                    sentiment_expressions.append('Aspekt ' + asp + ' wird im Text ' + TRANSLATE_POLARITIES_GER[re_labels[j]])
    
                sentiments = 'Anschließend ermitteln wir das Sentiment, welches gegenüber diesen Aspekten ausgedrückt wird: ' + 'Das ' + ', das '.join(sentiment_expressions[:-1]) + ' und das ' + sentiment_expressions[-1] + ' bewertet'
    
                labels_text = 'Das finale Ergebnis besteht somit aus den folgenden Aspekt-Sentiment-Paaren: '

            elif lang == 'en':
                aspect_labels = 'First, we identify all aspects addressed in the sentence: ' + ', '.join(re_aspects[:-1]) + ' and ' + re_aspects[-1]
                       
                sentiment_expressions = []
            
                for j, asp in enumerate(re_aspects):
                    sentiment_expressions.append('aspect ' + asp + ' is referenced ' + TRANSLATE_POLARITIES_EN[re_labels[j]])
    
                sentiments = 'Next, we determine the sentiment expressed towards these aspects: ' + 'The ' + ', the '.join(sentiment_expressions[:-1]) + ' and the ' + sentiment_expressions[-1] + ' in the text'
    
                labels_text = 'The final result thus consists of the following aspect-sentiment-pairs: '
                
        target_labels = '[' + ', '.join([f'({re_aspects[i]}, {re_labels[i]})' for i in range(len(re_aspects))]) + ']'

    elif absa_task == 'e2e' or absa_task == 'e2e-e':
        
        re_labels = examples_labels[1]
        re_phrases = examples_labels[0]

        if len(re_labels) == 1:
            if lang == 'ger':
                # If Aspect has Phrase
                raise NotImplementedError
                
            elif lang == 'en':
                sentiments = 'First, we identify the sentiment expressed in the sentence: The sentence expresses a ' + re_labels[0].lower() + ' sentiment'
                # If Aspect has Phrase
                if re_phrases[0] != 'NULL':
                    aspect_labels = 'Then, we determine what the target of the sentiment expression is: In this case, a ' + re_labels[0].lower() + ' sentiment is expressed towards the phrase \"' + re_phrases[0] + '\".' 
    
                # Aspect has no Phrase
                else:
                    aspect_labels = 'Then, we determine what the target of the sentiment expression is: Even though sentiment is expressed in the text, it is not directed towards an explicit opinion target. We thus assign its phrase the value \"NULL\"'
                labels_text = 'The final result thus consists of the following phrase-polarity-tuple: '

            target_labels = '[' + ', '.join([f'("{re_phrases[i]}", {re_labels[i]})' for i in range(len(re_phrases))]) + ']'

        elif len(re_labels) == 2: 
            if lang == 'ger':
                raise NotImplementedError

            elif lang == 'en':
                # If first but not second Aspect is without Phrase -> Swap Positions
                if re_phrases[0] == 'NULL' and not re_phrases[1] == 'NULL':
                    re_labels[0], re_labels[1] = re_labels[1], re_labels[0]
                    re_phrases[0], re_phrases[1] = re_phrases[1], re_phrases[0]
                    
                sentiments = 'First, we identify all sentiments expressed in the sentence: The sentence contains ' + re_labels[0].lower() + ' and ' + re_labels[1].lower() + ' sentiment'
                
                # If both Aspects are without Phrase
                if re_phrases[0] == 'NULL' and re_phrases[1] == 'NULL':
                    aspect_labels = 'Both sentiments are not directed towards an explicit opinion target, thus we assign the phrase \"NULL\" to both phrases'
                    
                # If only second Aspect is without Phrase
                elif re_phrases[1] == 'NULL':
                    aspect_labels = 'Then, we determine the opinion targets of the expressed sentiments: ' + re_labels[0].lower() + ' sentiment is directed towards \"' + re_phrases[0] + '\" while ' + re_labels[1].lower() + ' sentiment is expressed, but not directed towards an explicit opinion target. We thus assign its phrase the value \"NULL\"'
                    
                # Both Aspects have a Phrase
                else: 
                    aspect_labels = 'Then, we determine the opinion targets of the expressed sentiments: ' + re_labels[0].lower() + ' sentiment is directed towards \"' + re_phrases[0] + '\" and ' + re_labels[1].lower() + ' sentiment is directed towards \"' + re_phrases[1] + '\" in the text'
                labels_text = 'The final result thus consists of the following phrase-polarity-tuples: '

            target_labels = '[' + ', '.join([f'("{re_phrases[i]}", {re_labels[i]})' for i in range(len(re_phrases))]) + ']'
        
        elif len(re_labels) > 2: 

            aspects_wo_phrases = []
            labels_wo_phrases = []

            aspects_w_phrases = []
            labels_w_phrases = []
            phrases = []

            for i in range(len(re_labels)):

                # Accumulate aspects without phrases
                if re_phrases[i] == 'NULL':
                    labels_wo_phrases.append(re_labels[i])

                # Accumulate aspects with phrases
                else:
                    labels_w_phrases.append(re_labels[i])
                    phrases.append(re_phrases[i])

            aspect_labels = ''
            if lang == 'ger':
                raise NotImplementedError
                
            elif lang == 'en':
                labels_sorted = labels_w_phrases + labels_wo_phrases
                phrases_sorted = phrases + ['NULL' for _ in range(len(labels_wo_phrases))]
                
                sentiments = 'First, we identify all sentiments expressed in the sentence: The sentence contains ' + ', '.join([lab.lower() for lab in labels_sorted[:-1]]) + ' and ' + labels_sorted[-1].lower() + ' sentiments'
                
                if len(labels_w_phrases) == 1:
                    aspect_labels += 'Then, we determine the opinion targets of the expressed sentiments: ' + labels_w_phrases[0].lower() + ' sentiment is directed towards \"' + phrases[0] + '\"' 
                elif len(labels_w_phrases) == 2:
                    aspect_labels += 'Then, we determine the opinion targets of the expressed sentiments: ' + labels_w_phrases[0].lower() + ' sentiment is directed towards \"' + phrases[0] + '\" and ' + labels_w_phrases[1].lower() + ' sentiment is directed towards \"' + phrases[1] + '\"'
    
                elif len(labels_w_phrases) > 2:
                    aspect_labels += 'Then, we determine the opinion targets of the expressed sentiments: ' + labels_w_phrases[0].lower() + ' sentiment is directed towards \"' + phrases[0] + '\"'
    
                    for i in range(1, len(labels_w_phrases)-1):
                        aspect_labels += ', ' + labels_w_phrases[i].lower() + ' sentiment is directed towards \"' + phrases[i] + '\"'
    
                    aspect_labels += ' and ' +  labels_w_phrases[-1].lower() + ' sentiment is directed towards \"' + phrases[-1] + '\"'
    
                if len(labels_wo_phrases) > 0 and len(labels_wo_phrases) > 0:
                    aspect_labels += '. '
                    
                if len(labels_wo_phrases) == 1:
                    aspect_labels += 'Additionally, ' + labels_wo_phrases[0].lower() + ' sentiment is expressed in the sentence, but not directed towards an explicit opinion target. We thus assign its phrase the value \"NULL\"'
    
                elif len(labels_wo_phrases) == 2:
                    aspect_labels += 'Additionally, ' + labels_wo_phrases[0].lower() + ' and ' + labels_wo_phrases[1].lower() + ' sentiments are expressed in the sentence, but not directed towards an explicit opinion target. We thus assign their phrases the value \"NULL\"'
    
                elif len(labels_wo_phrases) > 2:
                    aspect_labels += 'Additionally, ' + ', '.join([lab.lower() for lab in labels_wo_phrases[:-1]]) + ' and ' + labels_wo_phrases[-1].lower() + ' sentiments are expressed in the sentence, but not directed towards an explicit opinion target. We thus assign their phrases the value \"NULL\"'
                
                labels_text = 'The final result thus consists of the following phrase-polarity-tuples: '
                
            target_labels = '[' + ', '.join([f'("{phrases_sorted[i]}", {labels_sorted[i]})' for i in range(len(phrases_sorted))]) + ']'
            
    elif absa_task == 'tasd':
        
        re_aspects = examples_labels[0]
        re_labels = examples_labels[1]
        re_phrases = examples_labels[2]

        if len(re_aspects) == 1:
            if lang == 'ger':
                # If Aspect has Phrase
                if re_phrases[0] != 'NULL':
                    aspect_labels = 'Zuerst identifizieren wir hierfür das im Satz angesprochene Aspekt und die dazugehörige Phrase: Das Aspekt ' + re_aspects[0] + ' wird mit der Phrase \"' + re_phrases[0] + '\" referenziert' 
    
                # Aspect has no Phrase
                else:
                    aspect_labels = 'Zuerst identifizieren wir hierfür das im Satz angesprochene Aspekt und die dazugehörige Phrase: Das Aspekt ' + re_aspects[0] + ' wird aus dem Zusammenhang des Satzes erschlossen und somit ohne Phrase referenziert. Daher weisen wir dessen Phrase den Wert \"NULL\" zu'
                sentiments = 'Anschließend ermitteln wir das Sentiment, welches gegenüber diesem Aspekt ausgedrückt wird: ' + 'Das Aspekt ' + re_aspects[0] + ' wird im Text ' + TRANSLATE_POLARITIES_GER[re_labels[0]] + ' bewertet'
                labels_text = 'Das finale Ergebnis besteht somit aus dem folgenden Aspekt-Sentiment-Phrasen-Tripel: '
                
            elif lang == 'en':
                # If Aspect has Phrase
                if re_phrases[0] != 'NULL':
                    aspect_labels = 'First, we identify the aspect addressed in the sentence and its corresponding phrase: The aspect ' + re_aspects[0] + ' is referenced with the phrase \"' + re_phrases[0] + '\"' 
    
                # Aspect has no Phrase
                else:
                    aspect_labels = 'First, we identify the aspect addressed in the sentence and its corresponding phrase: The aspect ' + re_aspects[0] + ' is inferred from the context of the sentence and is therefore referenced without a phrase. We thus assign its phrase the value \"NULL\"'
                sentiments = 'Next, we determine the sentiment expressed towards this aspect: ' + 'The aspect ' + re_aspects[0] + ' is mentioned ' + TRANSLATE_POLARITIES_EN[re_labels[0]] + ' in the text'
                labels_text = 'The final result thus consists of the following aspect-sentiment-phrase-triple: '

            target_labels = '[' + ', '.join([f'({re_aspects[i]}, {re_labels[i]}, "{re_phrases[i]}")' for i in range(len(re_aspects))]) + ']'
            
        elif len(re_aspects) == 2: 
            if lang == 'ger':
                # If second Aspect is without Phrase -> Swap Positions
                if re_phrases[0] == 'NULL' and not re_phrases[1] == 'NULL':
                    re_labels[0], re_labels[1] = re_labels[1], re_labels[0]
                    re_phrases[0], re_phrases[1] = re_phrases[1], re_phrases[0]
                    re_aspects[0], re_aspects[1] = re_aspects[1], re_aspects[0]
    
                # If second Aspect is without Phrase
                if re_phrases[0] == 'NULL' and re_phrases[1] == 'NULL':
                    aspect_labels = 'Zuerst identifizieren wir hierfür die im Satz angesprochenen Aspekte und deren dazugehörigen Phrasen: Die Aspekte ' + re_aspects[0] + ' und ' + re_aspects[1] + ' erschließen sich aus dem Zusammenhang des Satzes und werden somit ohne Phrase referenziert. Daher weisen wir beiden Aspekten die Phrase \"NULL\" zu'
                    
                # If first and second Aspect is without Phrase
                elif re_phrases[1] == 'NULL':
                    aspect_labels = 'Zuerst identifizieren wir hierfür die im Satz angesprochenen Aspekte und deren dazugehörigen Phrasen: Das Aspekt ' + re_aspects[0] + ' wird mit der Phrase \"' + re_phrases[0] + '\" referenziert, während sich das Aspekt ' + re_aspects[1] + ' aus dem Zusammenhang des Satzes erschließt und somit ohne Phrase referenziert wird. Daher weisen wir dessen Phrase den Wert \"NULL\" zu'
                    
                # Both Aspects have a Phrase
                else: 
                    aspect_labels = 'Zuerst identifizieren wir hierfür die im Satz angesprochenen Aspekte und deren dazugehörigen Phrasen: Das Aspekt ' + re_aspects[0] + ' wird mit der Phrase \"' + re_phrases[0] + '\" und ' + re_aspects[1] + ' wird mit der Phrase \"' + re_phrases[1] + '\" referenziert'
                sentiments = 'Anschließend ermitteln wir das Sentiment, welches gegenüber diesen Aspekten ausgedrückt wird: ' + 'Das Aspekt ' + re_aspects[0] + ' wird im Text ' + TRANSLATE_POLARITIES_GER[re_labels[0]] + ' und das Aspekt ' + re_aspects[1] + ' wird im Text ' + TRANSLATE_POLARITIES_GER[re_labels[1]] + ' bewertet'
                labels_text = 'Das finale Ergebnis besteht somit aus den folgenden Aspekt-Sentiment-Phrasen-Tripeln: '

            elif lang == 'en':
                # If second Aspect is without Phrase -> Swap Positions
                if re_phrases[0] == 'NULL' and not re_phrases[1] == 'NULL':
                    re_labels[0], re_labels[1] = re_labels[1], re_labels[0]
                    re_phrases[0], re_phrases[1] = re_phrases[1], re_phrases[0]
                    re_aspects[0], re_aspects[1] = re_aspects[1], re_aspects[0]
    
                # If second Aspect is without Phrase
                if re_phrases[0] == 'NULL' and re_phrases[1] == 'NULL':
                    aspect_labels = 'First, we identify all aspects addressed in the sentence and their corresponding phrases: The aspects ' + re_aspects[0] + ' und ' + re_aspects[1] + ' are inferred from the context of the sentence and are therefore referenced without a phrase. We thus assign the phrases \"NULL\" to both aspects'
                    
                # If first and second Aspect is without Phrase
                elif re_phrases[1] == 'NULL':
                    aspect_labels = 'First, we identify all aspects addressed in the sentence and their corresponding phrases: The aspect ' + re_aspects[0] + ' is referenced with the phrase \"' + re_phrases[0] + '\" while the aspect ' + re_aspects[1] + '  is inferred from the context of the sentence and is therefore referenced without a phrase. We thus assign its phrase the value \"NULL\"'
                    
                # Both Aspects have a Phrase
                else: 
                    aspect_labels = 'First, we identify all aspects addressed in the sentence and their corresponding phrases: The aspect ' + re_aspects[0] + ' is referenced with the phrase \"' + re_phrases[0] + '\" and the aspect ' + re_aspects[1] + ' is referenced with the phrase \"' + re_phrases[1] + '\" in the text'
                sentiments = 'Next, we determine the sentiment expressed towards these aspects: ' + 'The aspect ' + re_aspects[0] + ' is referenced ' + TRANSLATE_POLARITIES_EN[re_labels[0]] + ' and the aspect ' + re_aspects[1] + ' is mentioned ' + TRANSLATE_POLARITIES_EN[re_labels[1]] + ' in the text'
                labels_text = 'The final result thus consists of the following aspect-sentiment-phrase-triples: '
                
            target_labels = '[' + ', '.join([f'({re_aspects[i]}, {re_labels[i]}, "{re_phrases[i]}")' for i in range(len(re_aspects))]) + ']'
            
        elif len(re_aspects) > 2: 

            aspects_wo_phrases = []
            labels_wo_phrases = []

            aspects_w_phrases = []
            labels_w_phrases = []
            phrases = []

            for i in range(len(re_aspects)):

                # Accumulate aspects without phrases
                if re_phrases[i] == 'NULL':
                    aspects_wo_phrases.append(re_aspects[i])
                    labels_wo_phrases.append(re_labels[i])

                # Accumulate aspects with phrases
                else:
                    aspects_w_phrases.append(re_aspects[i])
                    labels_w_phrases.append(re_labels[i])
                    phrases.append(re_phrases[i])

            aspect_labels = ''
            if lang == 'ger':
                if len(aspects_w_phrases) == 1:
                    aspect_labels += 'Zuerst identifizieren wir hierfür das im Satz angesprochene Aspekt und die dazugehörige Phrase: Das Aspekt ' + aspects_w_phrases[0] + ' wird mit der Phrase \"' + phrases[0] + '\" referenziert' 
                elif len(aspects_w_phrases) == 2:
                    aspect_labels += 'Zuerst identifizieren wir hierfür die im Satz angesprochenen Aspekte und deren dazugehörigen Phrasen: Das Aspekt ' + aspects_w_phrases[0] + ' wird mit der Phrase \"' + phrases[0] + '\" und ' + aspects_w_phrases[1] + ' wird mit der Phrase \"' + phrases[1] + '\" referenziert'
    
                elif len(aspects_w_phrases) > 2:
                    aspect_labels += 'Zuerst identifizieren wir hierfür die im Satz angesprochenen Aspekte und deren dazugehörigen Phrasen: Das Aspekt ' + aspects_w_phrases[0] + ' wird mit der Phrase \"' + phrases[0] + '\"'
    
                    for i in range(1, len(aspects_w_phrases)-1):
                        aspect_labels += ', das Aspekt ' + aspects_w_phrases[i] + ' wird mit der Phrase \"' + phrases[i] + '\"'
    
                    aspect_labels += ' und das Aspekt ' +  aspects_w_phrases[-1] + ' wird mit der Phrase \"' + phrases[-1] + '\" referenziert'
    
                if len(aspects_w_phrases) > 0 and len(aspects_wo_phrases) > 0:
                    aspect_labels += '. '
                    
                if len(aspects_wo_phrases) == 1:
                    aspect_labels += 'Das Aspekt ' + aspects_wo_phrases[0] + ' wird aus dem Zusammenhang des Satzes erschlossen und somit ohne Phrase referenziert. Daher weisen wir der Phrase den Wert \"NULL\" zu'
    
                elif len(aspects_wo_phrases) == 2:
                    aspect_labels += 'Die Aspekte ' + aspects_wo_phrases[0] + ' und ' + aspects_wo_phrases[1] + ' erschließen sich aus dem Zusammenhang des Satzes und werden somit ohne Phrase referenziert. Daher weisen wir beiden Aspekten die Phrase \"NULL\" zu'
    
                elif len(aspects_wo_phrases) > 2:
                    aspect_labels += 'Die Aspekte ' + ', '.join(aspects_wo_phrases[:-1]) + ' und ' + aspects_wo_phrases[-1] + ' erschließen sich aus dem Zusammenhang des Satzes und werden somit ohne Phrase referenziert. Daher weisen wir diesen Aspekten die Phrase \"NULL\" zu'
                
                sentiment_expressions = []
    
                aspects_sorted = aspects_w_phrases + aspects_wo_phrases
                labels_sorted = labels_w_phrases + labels_wo_phrases
                phrases_sorted = phrases + ['NULL' for _ in range(len(labels_wo_phrases))]
                
                for j, asp in enumerate(aspects_sorted):
                    sentiment_expressions.append('Aspekt ' + asp + ' wird im Text ' + TRANSLATE_POLARITIES_GER[labels_sorted[j]])
    
                sentiments = 'Anschließend ermitteln wir das Sentiment, welches gegenüber diesen Aspekten ausgedrückt wird: ' + 'Das ' + ', das '.join(sentiment_expressions[:-1]) + ' und das ' + sentiment_expressions[-1] + ' bewertet'
    
                labels_text = 'Das finale Ergebnis besteht somit aus den folgenden Aspekt-Sentiment-Phrasen-Tripeln: '
            elif lang == 'en':
                if len(aspects_w_phrases) == 1:
                    aspect_labels += 'First, we identify the aspect addressed in the sentence and its corresponding phrase: The aspect ' + aspects_w_phrases[0] + ' is referenced with the phrase \"' + phrases[0] + '\"' 
                elif len(aspects_w_phrases) == 2:
                    aspect_labels += 'First, we identify all aspects addressed in the sentence and their corresponding phrases: The aspect ' + aspects_w_phrases[0] + ' is referenced with the phrase \"' + phrases[0] + '\" and ' + aspects_w_phrases[1] + ' is referenced with the phrase \"' + phrases[1] + '\"'
    
                elif len(aspects_w_phrases) > 2:
                    aspect_labels += 'First, we identify all aspects addressed in the sentence and their corresponding phrases: The aspect ' + aspects_w_phrases[0] + ' is referenced with the phrase \"' + phrases[0] + '\"'
    
                    for i in range(1, len(aspects_w_phrases)-1):
                        aspect_labels += ', the aspect ' + aspects_w_phrases[i] + ' is referenced with the phrase \"' + phrases[i] + '\"'
    
                    aspect_labels += ' and the aspect ' +  aspects_w_phrases[-1] + ' is referenced with the phrase \"' + phrases[-1] + '\"'
    
                if len(aspects_w_phrases) > 0 and len(aspects_wo_phrases) > 0:
                    aspect_labels += '. '
                    
                if len(aspects_wo_phrases) == 1:
                    aspect_labels += 'The aspect ' + aspects_wo_phrases[0] + ' is inferred from the context of the sentence and is therefore referenced without a phrase. We thus assign its phrase the value \"NULL\"'
    
                elif len(aspects_wo_phrases) == 2:
                    aspect_labels += 'The aspects ' + aspects_wo_phrases[0] + ' and ' + aspects_wo_phrases[1] + ' are inferred from the context of the sentence and are therefore referenced without a phrase. We thus assign their phrases the value \"NULL\"'
    
                elif len(aspects_wo_phrases) > 2:
                    aspect_labels += 'The aspects ' + ', '.join(aspects_wo_phrases[:-1]) + ' and ' + aspects_wo_phrases[-1] + ' are inferred from the context of the sentence and are therefore referenced without a phrase. We thus assign their phrases the value \"NULL\"'
                
                sentiment_expressions = []
    
                aspects_sorted = aspects_w_phrases + aspects_wo_phrases
                labels_sorted = labels_w_phrases + labels_wo_phrases
                phrases_sorted = phrases + ['NULL' for _ in range(len(labels_wo_phrases))]
                
                for j, asp in enumerate(aspects_sorted):
                    sentiment_expressions.append('aspect ' + asp + ' is mentioned ' + TRANSLATE_POLARITIES_EN[labels_sorted[j]])
    
                sentiments = 'Next, we determine the sentiment expressed towards these aspects: ' + 'The ' + ', the '.join(sentiment_expressions[:-1]) + ' and the ' + sentiment_expressions[-1] + ' in the text'
    
                labels_text = 'The final result thus consists of the following aspect-sentiment-phrase-triples: '       
                
            target_labels = '[' + ', '.join([f'({aspects_sorted[i]}, {labels_sorted[i]}, "{phrases_sorted[i]}")' for i in range(len(phrases_sorted))]) + ']'
            
    return few_shot_template.format(sent = examples_text, aspects = aspect_labels, sentiments = sentiments, labels_text = labels_text, labels = target_labels)

def createPromptText(lang, prompt_templates, prompt_style, example_text, example_labels, dataset_name = 'rest-16', absa_task = 'acd', train = False):

    # Set templates based on prompt config
    if dataset_name not in ['GERestaurant', 'rest-16']:
        raise NotImplementedError('Prompt template not found: Dataset name not valid.')
    else:
        dataset_name = dataset_name.replace('-','')[:6]

    if lang not in ['en', 'ger']:
        raise NotImplementedError('Prompt template not found: Prompt language not valid.')

    if absa_task not in ['acd', 'acsa', 'e2e', 'e2e-e', 'tasd']:
        raise NotImplementedError('Prompt template not found: Absa task not valid.')
    
    if prompt_style not in ['basic', 'context', 'cot']:
        raise NotImplementedError('Prompt template not found: Prompt style not valid.')
    else:
        template_prompt_style = prompt_style if prompt_style != 'cot' else 'context'

    try:
        prompt_template = prompt_templates[f'PROMPT_TEMPLATE_{lang.upper()}_{"E2E" if absa_task == "e2e-e" else absa_task.upper()}_{dataset_name.upper()}_{template_prompt_style.upper()}']
    except:
        raise NotImplementedError('Prompt template does not exist.')

    # Chain-of-Thought prompt format
    if prompt_style == 'cot':
        if absa_task == 'acd':
                raise NotImplementedError('Chain-of-Thought not implemented for ACD-Task')
        if lang == 'ger':
            if absa_task == 'acsa':
                example_template = 'Lass uns das Schritt für Schritt durchgehen. Wir möchten aus dem folgenden Satz alle Aspekt-Sentiment-Paare extrahieren: "{sent}". {aspects}. {sentiments}. \n{labels_text}{labels}'
            elif absa_task == 'e2e' or absa_task == 'e2e-e':
                example_template = 'Lets do this step by step. We would like to extract all opinion-target-phrase-sentiment-pairs from the following sentence: "{sent}". {aspects}. {sentiments}. \n{labels_text}{labels}'
            elif absa_task == 'tasd':
                example_template = 'Lass uns das Schritt für Schritt durchgehen. Wir möchten aus dem folgenden Satz alle Aspekt-Sentiment-Phrasen-Tripel extrahieren: "{sent}". {aspects}. {sentiments}. \n{labels_text}{labels}'
        elif lang == 'en':
            if absa_task == 'acsa':
                example_template = 'Lets do this step by step. We would like to extract all aspect-sentiment-pairs from the following sentence: "{sent}". {aspects}. {sentiments}. \n{labels_text}{labels}'
            elif absa_task == 'e2e' or absa_task == 'e2e-e':
                example_template = 'Lets do this step by step. We would like to extract all opinion-target-phrase-sentiment-tuples from the following sentence: "{sent}". {sentiments}. {aspects}. \n{labels_text}{labels}'
            elif absa_task == 'tasd':
                example_template = 'Lets do this step by step. We would like to extract all aspect-sentiment-phrase-triples from the following sentence: "{sent}". {aspects}. {sentiments}. \n{labels_text}{labels}\n\n'
    
    

    if example_labels is not None:
        try:
            reg_asp = re.compile(REGEX_ASPECTS_ACSD)
            reg_lab = re.compile(REGEX_LABELS_ACSD)
            reg_phr = re.compile(REGEX_PHRASES_ACSD)
            
            # Extract Aspects from  sample
            re_aspects = [reg_asp.match(pair)[1] for pair in example_labels]
            
            # Extract Polarities from  sample
            re_labels = [reg_lab.match(pair)[1] for pair in example_labels]
        
            # Extract Aspect-Phrases from  sample
            re_phrases = [reg_phr.match(pair)[1] for pair in example_labels]
            
        except:
            try:
                reg_asp = re.compile(REGEX_ASPECTS)
                reg_lab = re.compile(REGEX_LABELS)
                
                # Extract Aspects from  sample
                re_aspects = [reg_asp.match(pair)[1] for pair in example_labels]
                
                # Extract Polarities from  sample
                re_labels = [reg_lab.match(pair)[1] for pair in example_labels]
        
            except:
                raise NotImplementedError("Data-Format is not ['(Aspect Category, Sentiment Polarity, Aspect Phrase)', '(Aspect Category, Sentiment Polarity, Aspect Phrase)']")
        
        if absa_task == 'acd':        
            example_labels = re_aspects
    
        elif absa_task == 'acsa':
            example_labels = [re_aspects, re_labels]
            
        elif absa_task == 'e2e' or absa_task == 'e2e-e':
            example_labels = [re_phrases, re_labels]
            
        elif absa_task == 'tasd':
            example_labels = [re_aspects, re_labels, re_phrases]

    prompt = '### Instruction:\n' + prompt_template[0] + ' ' + prompt_template[1] + '\n\n' + f'### Input:\n{example_text} \n\n### Output:\n' 

    # Determine if train or test prompt and append target label if necessary

    if absa_task == 'acd':
        target_text = '[' + ', '.join(example_labels) + ']'
    elif absa_task == 'acsa':
         target_text = '[' + ', '.join([f'({example_labels[0][i]}, {example_labels[1][i]})' for i in range(len(example_labels[0]))]) + ']'
    elif absa_task == 'e2e' or absa_task == 'e2e-e':
         target_text = '[' + ', '.join([f'("{example_labels[0][i]}", {example_labels[1][i]})' for i in range(len(example_labels[0]))]) + ']'
    elif absa_task == 'tasd':
         target_text = '[' + ', '.join([f'({example_labels[0][i]}, {example_labels[1][i]}, "{example_labels[2][i]}")' for i in range(len(example_labels[0]))]) + ']'

    if train:
        if prompt_style == 'cot':
            return prompt + createCoTText(example_template, absa_task, example_text, example_labels, lang), None
        else:
            return prompt + target_text, None
    else:
        return prompt, target_text
        
def createPrompts(df_train, df_test, args, eos_token = ''):

    prompts_train = []
    prompts_test = []
    ground_truth_labels = []
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.dataset in ['rest-16', 'GERestaurant']:
        with open(os.path.join(base_dir, f'prompts_{args.dataset.replace("-","")}.json'), encoding='utf-8') as json_prompts:
             prompt_templates = json.load(json_prompts)
    else:
        raise NotImplementedError('Prompt-Template not found: File does not exist.')

    label_column = 'labels_phrases' if 'labels_phrases' in df_train.columns else 'labels' if 'labels' in df_train.columns else lambda x: raise_err(NotImplementedError('Dataset does not have column with label targets.'))

    for index, row in df_train.iterrows():       

        prompt, _ = createPromptText(lang = args.lang, prompt_templates = prompt_templates, prompt_style = args.prompt_style, example_text = row['text'], example_labels = row[label_column], dataset_name = args.dataset, absa_task = args.task, train = True) 

        prompts_train.append(prompt + eos_token)

    for index, row in df_test.iterrows():
        prompt, targets = createPromptText(lang = args.lang, prompt_templates = prompt_templates, prompt_style = args.prompt_style, example_text = row['text'], example_labels = row[label_column], dataset_name = args.dataset, absa_task = args.task)
        
        prompts_test.append(prompt + eos_token)
        ground_truth_labels.append(targets)
        
    return prompts_train, prompts_test, ground_truth_labels