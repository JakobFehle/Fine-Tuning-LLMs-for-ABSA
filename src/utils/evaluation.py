import re
import torch

from transformers import StoppingCriteria

REGEX_ASPECTS_ACD = r'\[([^\]]+)\]'
REGEX_ASPECTS_ACSD = r"\(([^,]+),[^,]+,\s*\"[^\"]*\"\)"
REGEX_LABELS_ACSD = r"\([^,]+,\s*([^,]+)\s*,\s*\"[^\"]*\"\s*\)"
REGEX_PHRASES_ACSD = r"\([^,]+,\s*[^,]+\s*,\s*\"([^\"]*)\"\s*\)"
REGEX_LABELS_ACSA = r'\(([^,]+),\s*([^)]+)\)'
REGEX_PAIRS_ACSA_ACSD = r'\([^()]+?\)'
POLARITIES = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

def sortCheckpoints(folders):
    model_config = folders[0].split('checkpoint-')[0]
    sorted_epochs = sorted(int(f.split('-')[-1]) for f in folders)

    return [f"{model_config}checkpoint-{epoch}" for epoch in sorted_epochs]

def safe_recursive_pattern(depth, max_depth):
    quoted_content = r'"(?:[^"\\]|\\.)*"'  # Matches anything inside quotes.
    if depth >= max_depth:
        return rf'(?:{quoted_content}|[^()])*'
    return rf'\((?:{quoted_content}|[^()]|{safe_recursive_pattern(depth + 1, max_depth)})*\)'

def extract_valid_e2e_tuples(text):
    # Define the pattern for a well-formed tuple: ("Phrase", Label)
    pattern = r'\(\s*"([^"]*)"\s*,\s*(POSITIVE|NEGATIVE|NEUTRAL)\s*\)'
    
    # Compile the regex to extract valid tuples
    compiled_pattern = re.compile(pattern)
    
    # Extract all matches from the string
    valid_tuples = compiled_pattern.findall(text)
    
    # Return the tuples in the format [('Phrase', 'Label'), ...]
    return valid_tuples

def extractAspects(output, task, cot = False, evaluation = False):
    def strip_cot_output(output, keywords):
        for keyword in keywords:
            if keyword in output:
                return output.split(keyword)[-1]
        return output

    # Validate Output
    if output.count('(') != output.count(')'):
        return []

    if cot and evaluation:
        keywords = [
            'folgenden Aspekt-Sentiment-Paar:', 'folgenden Aspekt-Sentiment-Paaren:',
            'the following aspect-sentiment-pair:', 'the following aspect-sentiment-pairs:',
            'folgenden Aspekt-Sentiment-Phrasen-Tripeln:', 'folgenden Aspekt-Sentiment-Phrasen-Tripel:',
            'the following aspect-sentiment-phrase-triple:', 'the following aspect-sentiment-phrase-triples:',
            'the following phrase-polarity-tuple:','the following phrase-polarity-tuples:'
        ]     
        output = strip_cot_output(output, keywords)
        
    if task == 'acd':

        pattern_asp = re.compile(REGEX_ASPECTS_ACD)
        matches = pattern_asp.findall(output)
        
        return matches[0].split(', ') if matches else []
        
    elif task == 'acsa':
                
        pattern_pairs = re.compile(REGEX_PAIRS_ACSA_ACSD)
        pattern_lab = re.compile(REGEX_LABELS_ACSA)
        
        pairs = pattern_pairs.findall(output)
        
        return [[m[1], m[2]] for pair in pairs if (m := pattern_lab.search(pair))] or []

    elif task in ['e2e', 'e2e-e', 'tasd']:
        if task in ['e2e', 'e2e-e']:
            
            return extract_valid_e2e_tuples(output)
        
            # return [
            #     [pattern_phrase.search(pair)[1], pattern_pol.search(pair)[1]]
            #     for pair in pairs if pattern_phrase.search(pair) and pattern_pol.search(pair)
            # ]
        else:  # task == 'tasd'
            max_depth = 5
            pattern_targets = re.compile(safe_recursive_pattern(0, max_depth))
            pairs = pattern_targets.findall(output)
            
            pattern_asp = re.compile(REGEX_ASPECTS_ACSD)
            pattern_pol = re.compile(REGEX_LABELS_ACSD)
            pattern_phrase = re.compile(REGEX_PHRASES_ACSD)
            
            return [
                [pattern_asp.search(pair)[1], pattern_pol.search(pair)[1], pattern_phrase.search(pair)[1]]
                for pair in pairs if pattern_asp.search(pair) and pattern_pol.search(pair) and pattern_phrase.search(pair)
            ]
    
def convertLabels(labels, task, label_space):
    false_predictions = []
    conv_l = []
    label_space = sorted(set(lab.split(':')[0] for lab in label_space)) if task == 'acd' else label_space
    for sample in labels:
        conv_s = []
        for pair in sample:
            if task != 'acd':
                pair_str = ':'.join(label.replace('"', '').replace("'", "") for label in pair[:2])
            else:
                pair_str = pair
                
            if pair_str in label_space or task == 'e2e' or task == 'e2e-e':
                conv_s.append(':'.join([pair_str, pair[2]]) if task == 'tasd' else pair_str)
            else:
                false_predictions.append(pair_str)
        conv_l.append(conv_s)

    return conv_l, false_predictions

def calculateMetrics(predictions, ground_truths):
    tp, fp, fn = 0, 0, 0
    
    for pred, gold in zip(predictions, ground_truths):
        pred_copy, gold_copy = pred[:], gold[:]  # Work with copies to avoid modifying original lists
        
        # Calculate True Positives
        for label in pred:
            if label in gold_copy:
                tp += 1
                gold_copy.remove(label)  # Remove the matched label from the gold list
            else:
                fp += 1  # False Positive: label in pred but not in gold
        
        # Remaining items in gold are False Negatives
        fn += len(gold_copy)

    # Precision, recall, F1, and accuracy calculations
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0
    support = tp + fp + fn
    
    return {'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), 'accuracy': round(accuracy, 4), 'support': support}


def createResults(pred_labels, gold_labels, label_space, task):
    
    if task == 'acd':
        # Calculate Micro Metrics
        micro_asp = calculateMetrics(pred_labels, gold_labels)
        label_space_grouped = sorted(list(set([lab.split(':')[0] for lab in label_space])))
        
        # Metrics by Aspects disregarding Polarities
        metrics_asp = []
        for i in range(len(label_space_grouped)):
            pred_labels_subset = [[label for label in pred if label == label_space_grouped[i]] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if label == label_space_grouped[i]] for gold in gold_labels]
            metrics_asp.append({'aspect': label_space_grouped[i], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})
    
        # micro_asp = calculateMetrics([[label for label in pred] for pred in pred_labels], [[label for label in gold] for gold in gold_labels], task)
        
        macro_asp = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'accuracy': "",
                 'support': micro_asp['support']}
        
        result = {metr['aspect']:metr['metrics'] for metr in metrics_asp}
        result['Micro-AVG'] = micro_asp
        result['Macro-AVG'] = macro_asp

        return result, None, None, None, None
    
    elif task == 'acsa':
        
        # Calculate Micro Metrics
        micro_asp_pol = calculateMetrics(pred_labels, gold_labels)
    
        # label_space = list(set([label for labels in gold_labels for label in labels]))
        # label_space.sort()

        # Group Aspect labels together ignoring their polarity
        label_space_grouped = [[label for label in label_space if label.split(':')[0] ==  aspect] for aspect in sorted(list(set([lab.split(':')[0] for lab in label_space])))]
        
        # Metrics by Aspects disregarding Polarities
        metrics_asp = []
        for i in range(len(label_space_grouped)):
            pred_labels_subset = [[label.split(':')[0] for label in pred if label == label_space_grouped[i][0].split(':')[0]] for pred in pred_labels]
            gold_labels_subset = [[label.split(':')[0] for label in gold if label == label_space_grouped[i][0].split(':')[0]] for gold in gold_labels]
            metrics_asp.append({'aspect': label_space_grouped[i][0].split(':')[0], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})
    
        micro_asp = calculateMetrics([[label.split(':')[0] for label in pred] for pred in pred_labels], [[label.split(':')[0] for label in gold] for gold in gold_labels])
        
        macro_asp = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'accuracy': "",
                 'support': micro_asp['support']}
    
        # Metrics by Aspects but Polarities have to match
        metrics_asp_pol = []
        for i in range(len(label_space_grouped)):
            pred_labels_subset = [[label for label in pred if label in label_space_grouped[i]] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if label in label_space_grouped[i]] for gold in gold_labels]
            metrics_asp_pol.append({'aspect': label_space_grouped[i][0].split(':')[0], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})
    
        macro_asp_pol = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_asp_pol]) / (len(label_space_grouped)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_asp_pol]) / (len(label_space_grouped)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_asp_pol]) / (len(label_space_grouped)), 4), 
                 'accuracy': "",
                 'support': micro_asp_pol['support']}
    
        # Metrics by Aspect-Polarity-Pairs (Classifier Class-Labels)
        metrics_pairs = []
        for i in range(len(label_space)):
            pred_labels_subset = [[label for label in pred if label == label_space[i]] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if label == label_space[i]] for gold in gold_labels]
            metrics_pairs.append({'aspect': label_space[i], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})
        
        macro_pairs = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_pairs]) / (len(label_space)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_pairs]) / (len(label_space)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_pairs]) / (len(label_space)), 4), 
                 'accuracy': "",
                 'support': micro_asp_pol['support']}

        # Metrics by Polarities
        metrics_pol = []
        
        for i in range(len(POLARITIES)):
            pred_labels_subset = [[label for label in pred if (isinstance(label, str) and label.split(':')[1] == POLARITIES[i])] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if (isinstance(label, str) and label.split(':')[1] == POLARITIES[i])] for gold in gold_labels]
            metrics_pol.append({'polarity': POLARITIES[i], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})

        macro_pol = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'accuracy': "",
                 'support': micro_asp_pol['support']}
            
        result_asp = {metr['aspect']:metr['metrics'] for metr in metrics_asp}
        result_asp['Micro-AVG'] = micro_asp
        result_asp['Macro-AVG'] = macro_asp
    
        result_asp_pol = {metr['aspect']:metr['metrics'] for metr in metrics_asp_pol}
        result_asp_pol['Micro-AVG'] = micro_asp_pol
        result_asp_pol['Macro-AVG'] = macro_asp_pol
    
        result_pairs = {metr['aspect']:metr['metrics'] for metr in metrics_pairs}
        result_pairs['Micro-AVG'] = micro_asp_pol
        result_pairs['Macro-AVG'] = macro_pairs

        result_pol = {metr['polarity']:metr['metrics'] for metr in metrics_pol}
        result_pol['Micro-AVG'] = micro_asp_pol
        result_pol['Macro-AVG'] = macro_pol
        
        return result_asp, result_asp_pol, result_pairs, result_pol, None

    elif task == 'e2e' or task == 'e2e-e':
        micro_pairs = calculateMetrics(pred_labels, gold_labels)

        metrics_pol = []
        
        for i in range(len(POLARITIES)):
            pred_labels_subset = [[label for label in pred if (isinstance(label, str) and label.split(':')[1] == POLARITIES[i])] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if (isinstance(label, str) and label.split(':')[1] == POLARITIES[i])] for gold in gold_labels]
            metrics_pol.append({'polarity': POLARITIES[i], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})

        macro_pol = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'accuracy': "",
                 'support': micro_pairs['support']}

        result_pol = {metr['polarity']:metr['metrics'] for metr in metrics_pol}
        result_pol['Micro-AVG'] = micro_pairs
        result_pol['Macro-AVG'] = macro_pol

        return None, None, None, result_pol, None
    
    elif task == 'tasd':
        # Calculate Micro Metrics
    
        # label_space = list(set([label for labels in gold_labels for label in labels]))
        # label_space.sort()
        label_space_grouped = [[label for label in label_space if label.split(':')[0] ==  aspect] for aspect in sorted(list(set([lab.split(':')[0] for lab in label_space])))]
        
        # Metrics by Aspects disregarding Polarities
        metrics_asp = []
        for i in range(len(label_space_grouped)):
            pred_labels_subset = [[label.split(':')[0] for label in pred if label.split(':')[0] == label_space_grouped[i][0].split(':')[0]] for pred in pred_labels]
            gold_labels_subset = [[label.split(':')[0] for label in gold if label.split(':')[0] == label_space_grouped[i][0].split(':')[0]] for gold in gold_labels]
            metrics_asp.append({'aspect': label_space_grouped[i][0].split(':')[0], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})
    
        micro_asp = calculateMetrics([[label.split(':')[0] for label in pred] for pred in pred_labels], [[label.split(':')[0] for label in gold] for gold in gold_labels])
        
        macro_asp = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_asp]) / (len(label_space_grouped) / 3), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_asp]) / (len(label_space_grouped) / 3), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_asp]) / (len(label_space_grouped) / 3), 4), 
                 'accuracy': "",
                 'support': micro_asp['support']}
    
        # Metrics by Aspects but Polarities have to match
        metrics_asp_pol = []
        for i in range(len(label_space_grouped)):
            pred_labels_subset = [[':'.join(label.split(':')[:2]) for label in pred if ':'.join(label.split(':')[:2]) in label_space_grouped[i]] for pred in pred_labels]
            gold_labels_subset = [[':'.join(label.split(':')[:2]) for label in gold if ':'.join(label.split(':')[:2]) in label_space_grouped[i]] for gold in gold_labels]
            metrics_asp_pol.append({'aspect': label_space_grouped[i][0].split(':')[0], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})

        micro_asp_pol = calculateMetrics([[':'.join(label.split(':')[:2]) for label in pred] for pred in pred_labels], 
                                       [[':'.join(label.split(':')[:2]) for label in gold] for gold in gold_labels])
        
        macro_asp_pol = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_asp_pol]) / (len(label_space_grouped)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_asp_pol]) / (len(label_space_grouped)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_asp_pol]) / (len(label_space_grouped)), 4), 
                 'accuracy': "",
                 'support': micro_asp_pol['support']}
    
        # Metrics by Aspect-Polarity-Pairs (Classifier Class-Labels)
        metrics_pairs = []
        for i in range(len(label_space)):
            pred_labels_subset = [[':'.join(label.split(':')[:2]) for label in pred if ':'.join(label.split(':')[:2]) == label_space[i]] for pred in pred_labels]
            gold_labels_subset = [[':'.join(label.split(':')[:2]) for label in gold if ':'.join(label.split(':')[:2]) == label_space[i]] for gold in gold_labels]
            metrics_pairs.append({'aspect': label_space[i], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})
        
        macro_pairs = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_pairs]) / (len(label_space)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_pairs]) / (len(label_space)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_pairs]) / (len(label_space)), 4), 
                 'accuracy': "",
                 'support': micro_asp_pol['support']}

        # Metrcis by Aspects but Polarties and Phrases have to match
        metrics_phrases = []
        for i in range(len(label_space_grouped)):
            pred_labels_subset = [[label for label in pred if ':'.join(label.split(':')[:2]) in label_space_grouped[i]] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if ':'.join(label.split(':')[:2]) in label_space_grouped[i]] for gold in gold_labels]
            metrics_phrases.append({'aspect': label_space_grouped[i][0].split(':')[0], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})

        micro_phrases = calculateMetrics([[label for label in pred] for pred in pred_labels], 
                                         [[label for label in gold] for gold in gold_labels])
        
        macro_phrases = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_phrases]) / (len(label_space_grouped)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_phrases]) / (len(label_space_grouped)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_phrases]) / (len(label_space_grouped)), 4), 
                 'accuracy': "",
                 'support': micro_phrases['support']}

        # Metrics by Polarities
        metrics_pol = []
        
        for i in range(len(POLARITIES)):
            pred_labels_subset = [[label for label in pred if (isinstance(label, str) and label.split(':')[1] == POLARITIES[i])] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if (isinstance(label, str) and label.split(':')[1] == POLARITIES[i])] for gold in gold_labels]
            metrics_pol.append({'polarity': POLARITIES[i], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})

        macro_pol = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'accuracy': "",
                 'support': micro_asp_pol['support']}
        
        result_asp = {metr['aspect']:metr['metrics'] for metr in metrics_asp}
        result_asp['Micro-AVG'] = micro_asp
        result_asp['Macro-AVG'] = macro_asp
    
        result_asp_pol = {metr['aspect']:metr['metrics'] for metr in metrics_asp_pol}
        result_asp_pol['Micro-AVG'] = micro_asp_pol
        result_asp_pol['Macro-AVG'] = macro_asp_pol
    
        result_pairs = {metr['aspect']:metr['metrics'] for metr in metrics_pairs}
        result_pairs['Micro-AVG'] = micro_asp_pol
        result_pairs['Macro-AVG'] = macro_pairs

        result_asp_pol_phrases = {metr['aspect']:metr['metrics'] for metr in metrics_phrases}
        result_asp_pol_phrases['Micro-AVG'] = micro_phrases
        result_asp_pol_phrases['Macro-AVG'] = macro_phrases

        result_pol = {metr['polarity']:metr['metrics'] for metr in metrics_pol}
        result_pol['Micro-AVG'] = micro_asp_pol
        result_pol['Macro-AVG'] = macro_pol
        
        return result_asp, result_asp_pol, result_pairs, result_pol, result_asp_pol_phrases