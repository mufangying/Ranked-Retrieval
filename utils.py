import math
import random
import numpy as np
from scipy.stats import binom
from tqdm import tqdm

random.seed(2025)

# load files
def load_data(file_path):
    data = {}
    all_scores = []
    with open(file_path) as f:
        for idx, line in enumerate(f):
            line_data = line.strip().split(',') 
            query_id = line_data[0]
            doc_id = int(line_data[1])
            label = int(line_data[2])
            score = float(line_data[3])
            all_scores.append(score)
            doc_data = (doc_id, label, score)
    
            if query_id not in data.keys():
                data[query_id] = list()
            data[query_id].append(doc_data)

    # sort the list so that they arrange in the descending order of their relevance scores
    for query_id in data.keys():
        data[query_id] = sorted(data[query_id], key=lambda x: x[1], reverse=True)

    return data, all_scores


# check all the relevance levels
def get_relevance_levels(file_path):
    all_scores = []
    with open(file_path) as f:
        for idx, line in enumerate(f):
            line_data = line.strip().split(',') 
            label = int(line_data[2])
            all_scores.append(label)

    return  set(all_scores)


# exclude non-relevant quenries
def exclude_ids(data):
    exclude_ids = []
    for query_id in data.keys():
        label_set = set([d[1] for d in data[query_id]])
        if len(label_set) == 1 and label_set == {0}:
        # if 1 not in set([d[1] for d in data[query_id]]):   ## alternatively, if the label is binary, we can check if 1 is in the set.
            exclude_ids.append(query_id)
    for query_id in exclude_ids:
        del data[query_id]
    print(len(exclude_ids))
    return data

# zip l1 data and l2 data
def zip_data(l1_data, l2_data, ids):
    zipped_data = {}
    for id in ids:
        zipped_data[id] = (l1_data[id], l2_data[id])
    return zipped_data




# split data into caliration data and test data
def split_query_ids(data, split_ratio=0.5):
    query_ids = list(data.keys())
    num_queries = len(query_ids)
    num_val = int (num_queries * split_ratio)
    val_idx = set(random.sample(range(num_queries), num_val))
    val_ids = [query_ids[idx] for idx in val_idx]
    test_ids = [query_ids[idx] for idx in range(num_queries) if idx not in val_idx]
    return val_ids, test_ids

# split calibration data into two parts with equal size
def split_val_data(val_data):
    val_ids = list(val_data.keys())
    random.shuffle(val_ids)
    split_index = len(val_ids) // 2
    val_ids_1 = val_ids[:split_index]
    val_ids_2 = val_ids[split_index:]

    val_data_1 = {id: val_data[id] for id in val_ids_1}
    val_data_2 = {id: val_data[id] for id in val_ids_2}
    return (val_data_1, val_data_2)


# first stage risk computation
def calc_l1_risk_for_query(docs_for_query, threshold, relevance_level=1):
    ground_truth_docs = set([doc[0] for doc in docs_for_query if doc[1] >= relevance_level])
    fetched_docs = set([doc[0] for doc in docs_for_query if doc[2] >= 1 - threshold])

    # retrieved relevant documents
    num_fetched = len(ground_truth_docs.intersection(fetched_docs))
    loss = 1 - num_fetched / (1.0 if len(ground_truth_docs) == 0 else len(ground_truth_docs))

    return (loss, fetched_docs)



# first stage lambda computation
def calc_retrieval_lambda_1(val_data,  alpha, relevance_level = 1):
    
    precision = 0.001
    low_lambda = 0
    high_lambda = 1
    M = len(val_data.keys())
    threshold = (M + 1) * alpha - 1
    
    # check
    assert (M + 1) * alpha - 1 > 0, "No solution!"
    
    while high_lambda - low_lambda > precision + 1e-6:
        
        mid_lambda = (low_lambda+high_lambda)/2
        mid_lambda = round(mid_lambda, 3)
        
        total_loss = 0 
        for query_id, (docs_for_query, _)  in val_data.items():
            total_loss += calc_l1_risk_for_query(docs_for_query, mid_lambda, relevance_level)[0]
        
        if total_loss > threshold:
            low_lambda = mid_lambda
        elif total_loss < threshold:
            high_lambda = mid_lambda
        else:
            high_lambda = mid_lambda
        
    return high_lambda 


# second stage risk computation
def calc_l2_risk_for_query(l1_docs_for_query, l2_docs_for_query, lambda_val, gamma, relevance_level = 1):
    
    l1_retrieved_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= 1 - lambda_val])
    l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= 1 - gamma and doc[0] in l1_retrieved_docs])
    
    denominator = 0
    numerator = 0
    # print(f'Length of l2_docs_for_query: {len(l2_docs_for_query)}')
    for i in range(len(l2_docs_for_query)):
        doc = l2_docs_for_query[i]
        if doc[1] >= relevance_level:
            denominator += 1/math.log(i+2)
            numerator += 1/math.log(i+2)*(doc[0] in l2_retained_docs)
        
    if denominator == 0:
        return 1
    else:
        return 1 - numerator / denominator
    


# second stage lambda computation
def calc_retrieval_lambda_2(val_data, beta, relevance_level = 1):
    high_lambda = 1
    low_lambda = 0
    precision = 0.001
    
    M = len(val_data.keys())
    threshold = (M + 1) * beta - 1
    assert (M + 1) * beta - 1 > 0, "No solution!"
    
    
    while high_lambda - low_lambda > precision + 1e-6:
        
        mid_lambda = (low_lambda+high_lambda)/2
        mid_lambda = round(mid_lambda,3)
        
        total_loss = 0 
        for query_id, (l1_docs_for_query, l2_docs_for_query)  in val_data.items():
            total_loss += calc_l2_risk_for_query(l1_docs_for_query, l2_docs_for_query, 
                                                     mid_lambda, 1, relevance_level)
            
        if total_loss > threshold:
            low_lambda = mid_lambda
        elif total_loss < threshold:
            high_lambda = mid_lambda
        else:
            high_lambda = mid_lambda
  
    return high_lambda 





# second stage gamma computation
def calc_rank_gamma(val_data, beta, lambda_val, relevance_level = 1):
    high_gamma= 1
    low_gamma = 0
    precision = 0.001
    
    M = len(val_data.keys())
    threshold = (M + 1) * beta - 1
    assert (M + 1) * beta - 1 > 0, "No solution!"
    
    
    while high_gamma - low_gamma > precision + 1e-6:
        
        mid_gamma = (low_gamma+high_gamma)/2
        mid_gamma = round(mid_gamma, 3)
        
        total_loss = 0 
        for query_id, (l1_docs_for_query, l2_docs_for_query)  in val_data.items():
            total_loss += calc_l2_risk_for_query(l1_docs_for_query, l2_docs_for_query, 
                                                    lambda_val,  mid_gamma, relevance_level)
            
        if total_loss > threshold:
            low_gamma = mid_gamma
        elif total_loss < threshold:
            high_gamma = mid_gamma
        else:
            high_gamma = mid_gamma
  
    return high_gamma 


# parameter selection without data splitting
def full_cal_parameter_selection(val_data, alpha, beta):
    
    # compute lambda
    lambda_1 = calc_retrieval_lambda_1(val_data,  alpha)
    lambda_2 = calc_retrieval_lambda_2(val_data,  beta)
    lower_lambda = max(lambda_1, lambda_2)
    
    # define searching grid
    lambda_grid = np.arange(lower_lambda, 1 + 0.0005, 0.001)
    
    # parameter selection through loss1: size in stage 2
    best_size = 1e8
    best_gamma = 1
    best_lambda = 1
    
    for i in range(len(lambda_grid)):
        lambda_val = lambda_grid[i]
        
        # find gamma in stage 2
        gamma = calc_rank_gamma(val_data, beta, lambda_val)
     
        total_l2_size = 0
        for _, (l1_docs_for_query, l2_docs_for_query) in val_data.items():
                l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= 1 - lambda_val])
                l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= 1 - gamma and doc[0] in l1_fetched_docs])
                total_l2_size += len(l2_retained_docs)
        
        # convert to mean prediction set sizes
        total_l2_size = total_l2_size/len(val_data.keys())
        

        if total_l2_size < best_size:
            best_size = total_l2_size
            best_gamma = gamma
            best_lambda = lambda_val
        
        
    return   best_lambda, best_gamma
    

# parameter selection with data splitting
def split_cal_parameter_selection(val_data_1, val_data_2, alpha, beta):
    
    # compute lambda
    lambda_1 = calc_retrieval_lambda_1(val_data_1,  alpha)
    
    # use val_data1 to estimate lambda0
    lambda_2 = calc_retrieval_lambda_2(val_data_1,  beta)
    lambda_val = max(lambda_1, lambda_2)
    
    # compute gamma
    gamma = calc_rank_gamma(val_data_2, beta, lambda_val, relevance_level = 1)
     
    return  lambda_val, gamma


###  LTT framework implementation - Bonferroni + fixed sequence design

# function to compute p-values
def hb_p_value(r_hat, n, alpha):
    bentkus_p_value = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)
    
    def h1(y, mu, eps=1e-10):
        y = np.clip(y, eps, 1 - eps)
        mu = np.clip(mu, eps, 1 - eps)
        with np.errstate(divide='ignore', invalid='ignore'):
            return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))
    
    hoeffding_p_value = np.exp(-n * h1(min(r_hat, alpha), alpha))
    return min(bentkus_p_value, hoeffding_p_value)



# compute p-values in the first stage
def get_pvalue_stage1(val_data, alpha, lam, relevance_level = 1):
    
    total_loss = 0
    nq = len(val_data.keys())
    for query_id, (docs_for_query, _)  in val_data.items():
        total_loss += calc_l1_risk_for_query(docs_for_query, lam, relevance_level)[0]
    risk_hat = total_loss/nq
    
    return hb_p_value(risk_hat, nq, alpha)


    
# parameter selection through lam_seq and gamma_seq
def ltt_parameter_selection(val_data, delta, alpha, beta, lam_seq, gamma_seq, relevance_level = 1):
    # lam_seq, gamma_seq are both python lists: descending order
    
    # Bonferroni's correction
    delta1 = delta/len(lam_seq)
    
    # compute p-values
    p1_list = [get_pvalue_stage1(val_data, alpha, lam_seq[i]) for i in range(len(lam_seq))]
    
    # for selected lambda, compute feasible gamma
    lam_selected = [lam_seq[i] for i in range(len(lam_seq)) if p1_list[i] < delta1 ]
    
    # parameter selection based on prediction set size in stage 2
    best_size = 1e8
    best_gamma = 1
    best_lambda = 1
    
    for i in range(len(lam_selected)):
        
        lambda_val = lam_selected[i]
        j = 0
        p_value2 = 0
        while j < len(gamma_seq) and p_value2 < delta1:
            gamma = gamma_seq[j]
            
            total_loss = 0
            total_l2_size = 0
            for _, (l1_docs_for_query, l2_docs_for_query) in val_data.items():
                    l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= 1 - lambda_val])
                    l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= 1 - gamma and doc[0] in l1_fetched_docs])
                    total_l2_size += len(l2_retained_docs)
                    total_loss += calc_l2_risk_for_query(l1_docs_for_query, l2_docs_for_query, 
                                                     lambda_val, gamma, relevance_level)
            # convert to mean prediction set sizes
            total_l2_size = total_l2_size/len(val_data.keys())
            total_loss = total_loss/len(val_data.keys())
            p_value2 = hb_p_value(total_loss, len(val_data.keys()), beta)
            j+=1 
            if p_value2 >= delta1:
                break

            if total_l2_size < best_size:
                best_size = total_l2_size
                best_gamma = gamma
                best_lambda = lambda_val
        # print(f'(lambda,gamma)  = { lambda_val, gamma} \n')
        
    return best_lambda, best_gamma



# function to evaluate test data for mslr dataset
def evaluate_test(test_data, lambda_val, gamma, relevance_level = 1):
    
    # compute first stage and second stage average risk
    l1_total_loss = 0
    l2_total_loss = 0
    total_l2_size = 0
    recall1 = []
    recall2 = []
    precision = []
    
    
    for query_id, (l1_docs_for_query, l2_docs_for_query)  in test_data.items():
        l1_total_loss += calc_l1_risk_for_query(l1_docs_for_query, lambda_val, relevance_level)[0]
        l2_total_loss += calc_l2_risk_for_query(l1_docs_for_query, l2_docs_for_query, 
                                                    lambda_val, gamma, relevance_level)
        l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= 1 - lambda_val])
        l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= 1 - gamma and doc[0] in l1_fetched_docs])
        total_l2_size += len(l2_retained_docs)
        
        
        # recall2
        retrieved_relevant_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= 2 and doc[0] in l2_retained_docs])
        relevant_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= 2])
        if len(relevant_docs) > 0:
            recall2.append(  len( retrieved_relevant_docs)/ len(relevant_docs) )
        else: 
            recall2.append( np.nan)
        
        # recall1
        retrieved_relevant_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] == 1 and doc[0] in l2_retained_docs])
        relevant_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] == 1])
        if len(relevant_docs) > 0:
            recall1.append(  len( retrieved_relevant_docs)/ len(relevant_docs) )
        else: 
            recall1.append( np.nan)


        # compute precision
        retrieved_relevant_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= 1 and doc[0] in l2_retained_docs])
        if len(l2_retained_docs) >0:
            precision.append( len(retrieved_relevant_docs) / len(l2_retained_docs) )
        else:
            precision.append(np.nan)
            # print(f'Precision is NA for query: {query_id}\n')
            # print(f'lambda: {lambda_val}, gamma: {gamma} \n')

        
    
    
    return np.array([lambda_val, gamma, round( l1_total_loss/len(test_data.keys()),3), round(l2_total_loss/len(test_data.keys()),3),
                    round( total_l2_size/len(test_data.keys()),3),  
                    round(np.nanmean(recall2),3), round(np.nanmean(recall1),3), round(np.nanmean(precision),3) ])



# function to run experiments multiple times
def run_experiments(num_iter, l1_data, l2_data, delta, alpha, beta, lam_seq, gamma_seq, relevance_level = 1):    
    # initialize matrix to store results
    # 3 methods
    # columns:  lambda, gamma, risk in the first stage, risk in the second stage, 
    #           prediction set size in the second stage, avg. recall,  avg. precision, avg. F1
    results = np.zeros((3, num_iter, 8))
    for i in tqdm(range(num_iter)):
        # data splitting
        val_ids, test_ids = split_query_ids(l1_data, 0.5)
        val_zipped_data = zip_data(l1_data, l2_data, val_ids)
        val_zipped_data_1, val_zipped_data_2 = split_val_data(val_zipped_data)
        test_zipped_data = zip_data(l1_data, l2_data, test_ids)
        
        # paremeter selection through different methods
        lambda_val, gamma = full_cal_parameter_selection(val_zipped_data, alpha, beta)
        results[0,i,:] = evaluate_test(test_zipped_data, lambda_val, gamma)
        
        
        lambda_val, gamma = split_cal_parameter_selection(val_zipped_data_1, val_zipped_data_2, alpha, beta)
        results[1,i,:] = evaluate_test(test_zipped_data, lambda_val, gamma)
        
        lambda_val, gamma = ltt_parameter_selection(val_zipped_data, delta, alpha, beta, 
                                                    lam_seq, gamma_seq)
        results[2,i,:] = evaluate_test(test_zipped_data, lambda_val, gamma)
        
    return np.mean(results, axis = 1)



# vary beta
def run_experiments_varying_beta(num_iter, l1_data, l2_data, delta, alpha, beta_seq, lam_seq, gamma_seq, relevance_level = 1):    
    # initialize matrix to store results
    # 3 methods
    # columns:  lambda, gamma, risk in the first stage, risk in the second stage, 
    #           prediction set size in the second stage, avg. recall,  avg. precision, avg. F1
    
    results = np.zeros((len(beta_seq), 3, 8))
    
    for i in range(len(beta_seq)):
        results[i,:,:] = run_experiments(num_iter, l1_data, l2_data, delta, alpha, beta_seq[i], lam_seq, gamma_seq)
    
    return results



# function to evaluate test data for yahoo/ms dataset
def evaluate_test_ms(test_data, lambda_val, gamma, relevance_level = 1):
    
    # compute first stage and second stage average risk
    l1_total_loss = 0
    l2_total_loss = 0
    total_l2_size = 0
    recall1 = []
    precision = []
    
    
    for query_id, (l1_docs_for_query, l2_docs_for_query)  in test_data.items():
        l1_total_loss += calc_l1_risk_for_query(l1_docs_for_query, lambda_val, relevance_level)[0]
        l2_total_loss += calc_l2_risk_for_query(l1_docs_for_query, l2_docs_for_query, 
                                                    lambda_val, gamma, relevance_level)
        l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= 1 - lambda_val])
        l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= 1 - gamma and doc[0] in l1_fetched_docs])
        total_l2_size += len(l2_retained_docs)
        
        
        
        # recall1
        retrieved_relevant_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= 1 and doc[0] in l2_retained_docs])
        relevant_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= 1])
        if len(relevant_docs) > 0:
            recall1.append(  len( retrieved_relevant_docs)/ len(relevant_docs) )
        else: 
            recall1.append( np.nan)



        # compute precision
        if len(l2_retained_docs) >0:
            precision.append( len(retrieved_relevant_docs) / len(l2_retained_docs) )
        else:
            precision.append(np.nan)
            # print('Precision is NA \n')
            # print(f'lambda: {lambda_val}, gamma: {gamma} \n')
        
    
    
    return np.array([lambda_val, gamma, round( l1_total_loss/len(test_data.keys()),3), round(l2_total_loss/len(test_data.keys()),3),
                    round( total_l2_size/len(test_data.keys()),3), 
                   round(np.nanmean(recall1),3), round(np.nanmean(precision),3) ])



# function to run experiments multiple times
def run_experiments_ms(num_iter, l1_data, l2_data, delta, alpha, beta, lam_seq, gamma_seq, relevance_level = 1):    
    # initialize matrix to store results
    # 3 methods
    # columns:  lambda, gamma, risk in the first stage, risk in the second stage, 
    #           prediction set size in the second stage, avg. recall,  avg. precision, avg. F1


   
    results = np.zeros((3, num_iter, 7))
    for i in tqdm(range(num_iter)):
        # data splitting
        val_ids, test_ids = split_query_ids(l1_data, 0.5)
        val_zipped_data = zip_data(l1_data, l2_data, val_ids)
        val_zipped_data_1, val_zipped_data_2 = split_val_data(val_zipped_data)
        test_zipped_data = zip_data(l1_data, l2_data, test_ids)
        
        # paremeter selection through different methods
        lambda_val, gamma = full_cal_parameter_selection(val_zipped_data, alpha, beta)
        lambda_val = round(lambda_val,2)
        gamma = round(gamma,2)
        results[0,i,:] = evaluate_test_ms(test_zipped_data, lambda_val, gamma)
        
        
        lambda_val, gamma = split_cal_parameter_selection(val_zipped_data_1, val_zipped_data_2, alpha, beta)
        lambda_val = round(lambda_val,2)
        gamma = round(gamma,2)
        results[1,i,:] = evaluate_test_ms(test_zipped_data, lambda_val, gamma)
        
        lambda_val, gamma = ltt_parameter_selection(val_zipped_data, delta, alpha, beta, 
                                                    lam_seq, gamma_seq)
        results[2,i,:] = evaluate_test_ms(test_zipped_data, lambda_val, gamma)
        
    return np.mean(results, axis = 1)



# vary beta
def run_experiments_varying_beta_ms(num_iter, l1_data, l2_data, delta, alpha, beta_seq, lam_seq, gamma_seq, relevance_level = 1):    
    # initialize matrix to store results
    # 3 methods
    # columns:  lambda, gamma, risk in the first stage, risk in the second stage, 
    #           prediction set size in the second stage, avg. recall,  avg. precision, avg. F1
    
    results = np.zeros((len(beta_seq), 3, 7))
    
    for i in range(len(beta_seq)):
        results[i,:,:] = run_experiments_ms(num_iter, l1_data, l2_data, delta, alpha, beta_seq[i], lam_seq, gamma_seq)
    
    return results

    
    

