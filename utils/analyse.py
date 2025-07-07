import numpy as np
import pandas as pd
import re
import ast

def compute_bias_metrics(cm, aggregate=None):
    s         = cm.sum()
    diag      = np.diag(cm)            
    c         = diag.sum()
    actual    = cm.sum(axis=1)         
    predicted = cm.sum(axis=0)         
    recall = np.where(actual>0, diag / actual, np.nan)

    tpr = diag / actual
    fpr = (predicted - diag) / (s - actual)
    tnr = 1 - fpr
    tpr = np.where(actual>0, tpr,   np.nan)
    fpr = np.where((s-actual)>0, fpr, np.nan)
    tnr = np.where((s-actual)>0, tnr, np.nan)


    # Accuracy
    accuracy = diag.sum() / s
    
    # Accuracy per group
    accuracy_per_group = []
    if aggregate is not None:
        for group in aggregate:
            group_ = list(set(group).intersection(set(range(cm.shape[0]))))
            if len(group_) == 0:
                accuracy_per_group.append([])
            else:
                tp_sum  = diag[group_].sum()
                act_sum = actual[group_].sum()
                accuracy_per_group.append(tp_sum / act_sum if act_sum>0 else np.nan)


    # Balanced accuracy 
    bac = np.nanmean((tpr + tnr) / 2)

    # G-mean
    if np.any(recall == 0):
        gmean = 0.0
    else:
        valid = recall[~np.isnan(recall)]
        gmean = np.prod(valid)**(1.0/len(valid))
     
    # MCC
    sum_t = np.dot(actual, actual)
    sum_p = np.dot(predicted, predicted)
    num   = c * s - np.dot(actual, predicted)
    den   = np.sqrt((s*s - sum_p) * (s*s - sum_t))
    mcc   = num / den if den>0 else np.nan

    # Disparate Impact
    di     = predicted / actual
    di     = np.where(actual>0, di, np.nan)
    di_std = np.sqrt(np.nanmean((di - 1)**2))

    # Equalized Odds variance
    mean_tpr = np.nanmean(tpr)
    mean_fpr = np.nanmean(fpr)
    eo_var   = np.nanmean(np.abs(tpr - mean_tpr) + np.abs(fpr - mean_fpr))

    # Bias Amplification
    p_act   = actual    / s
    p_pred  = predicted / s
    ba      = p_pred - p_act
    ba_mean = np.nanmean(np.abs(ba))

    # Confusion Entropy
    with np.errstate(divide='ignore', invalid='ignore'):
        probs = cm / actual[:, None]
    probs = np.nan_to_num(probs)
    h      = -np.sum(probs * np.log(probs + 1e-20), axis=1)
    ce_mean = np.nanmean(h)

    return {
        'accuracy': accuracy,
        'accuracy_per_group': accuracy_per_group,
        'bac': bac,
        'gmean': gmean,
        'mcc': mcc,
        'DI_std': di_std,
        'EO_var': eo_var,
        'BA_mean': ba_mean,
        'CE_mean': ce_mean,
        'DI_per_class': di,
        'TPR_per_class': tpr,
        'FPR_per_class': fpr,
        'BA_per_class': ba,
        'Confusion_entropy_per_class': h
    }

def metrics_series(cms, aggregate=None):
    records = []
    for cm in cms:
        m = compute_bias_metrics(cm, aggregate=aggregate)

        record = {
            'accuracy': m['accuracy'],
            'bac': m['bac'],
            'gmean': m['gmean'],
            'mcc': m['mcc'],
            'DI_std': m['DI_std'],
            'EO_var': m['EO_var'],
            'BA_mean': m['BA_mean'],
            'CE_mean': m['CE_mean'],
        }
        if aggregate is not None:
            for classes_idx, classes in enumerate(aggregate):
                classes_ = list(set(classes).intersection(set(range(m['DI_per_class'].shape[0]))))
                if len(classes_) == 0:
                    record[f'DI_{classes_idx}'] = np.nan
                    record[f'TPR_{classes_idx}'] = np.nan
                    record[f'FPR_{classes_idx}'] = np.nan
                    record[f'BA_{classes_idx}'] = np.nan
                    record[f'H_{classes_idx}'] = np.nan
                    record[f'accuracy_{classes_idx}'] = np.nan
                else:
                    record[f'DI_{classes_idx}'] = m['DI_per_class'][classes_].mean()
                    record[f'TPR_{classes_idx}'] = m['TPR_per_class'][classes_].mean()
                    record[f'FPR_{classes_idx}'] = m['FPR_per_class'][classes_].mean()
                    record[f'BA_{classes_idx}'] = m['BA_per_class'][classes_].mean()
                    record[f'H_{classes_idx}'] = m['Confusion_entropy_per_class'][classes_].mean()
                    record[f'accuracy_{classes_idx}'] = m['accuracy_per_group'][classes_idx] if m['accuracy_per_group'] else np.nan 
        records.append(record)
    df = pd.DataFrame.from_records(records)
    return df

def parse_log_tables(log_file_path):
    all_parsed_tables = []
    current_table_data = []
    in_table = False

    with open(log_file_path, 'r') as f:
        for line in f:
            if re.search(r'\[mos\.py\] => Class\s+Count\s+Acc\s+CorrectIterHist\s+WrongIterHist\s+WrongAdapters', line):
                in_table = True
                current_table_data = []
                continue

            if in_table and re.search(r'\[trainer\.py\] => No NME accuracy\.', line):
                in_table = False
                if current_table_data:
                    all_parsed_tables.append(current_table_data)
                continue

            if in_table:
                if '--------------------------------' in line:
                    continue

                match = re.search(
                    r'\[mos\.py\] =>\s+' # Match the log prefix and initial space
                    r'(\d+)\s+'           # Class
                    r'(\d+)\s+'           # Count
                    r'([\d.]+)\s+'        # Acc
                    r'(\{.*?\})\s+'      # CorrectIterHist (non-greedy)
                    r'(\{.*?\})\s*'      # WrongIterHist (non-greedy)
                    r'(\{.*?\})?',        # WrongAdapters (optional, non-greedy)
                    line
                )

                if match:
                    try:
                        class_val = int(match.group(1))
                        count_val = int(match.group(2))
                        acc_val = float(match.group(3))

                        correct_iter_hist_str = match.group(4).strip()
                        wrong_iter_hist_str = match.group(5).strip()
                        wrong_adapters_str = match.group(6)

                        # Handle potential None for optional last group
                        if wrong_adapters_str:
                            wrong_adapters_str = wrong_adapters_str.strip()
                        else:
                            wrong_adapters_str = "{}" # Default to empty dict if not found

                        row_dict = {
                            'Class': class_val,
                            'Count': count_val,
                            'Acc': acc_val,
                            'CorrectIterHist': ast.literal_eval(correct_iter_hist_str),
                            'WrongIterHist': ast.literal_eval(wrong_iter_hist_str),
                            'WrongAdapters': ast.literal_eval(wrong_adapters_str)
                        }
                        current_table_data.append(row_dict)
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing line: {line.strip()}")
                        print(f"Error details: {e}")
                        print(f"Problematic strings - CorrectIterHist: '{correct_iter_hist_str}', WrongIterHist: '{wrong_iter_hist_str}', WrongAdapters: '{wrong_adapters_str}'")
                        continue
    return all_parsed_tables

