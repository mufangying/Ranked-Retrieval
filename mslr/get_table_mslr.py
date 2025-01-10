from utils import *


# load data
l1_file_name = "MSLRWEB10K_L1.txt"
l2_file_name = "MSLRWEB10K_L2.txt"

l1_data, all_scores = load_data(l1_file_name)
l2_data, _ = load_data(l2_file_name)


delta = 0.01
lam_seq = [1 - 0.001*i for i in range(51)]
gamma_seq = [1 - 0.001*i for i in range(51)]
num_iter = 10


alpha, beta = 0.1, 0.1
print(f'alpha: {alpha}, beta: {beta} \n')
print( run_experiments(num_iter, l1_data, l2_data, delta, alpha, beta, lam_seq, gamma_seq, relevance_level = 1) )
print('\n')


alpha, beta = 0.01, 0.1
print(f'alpha: {alpha}, beta: {beta} \n')
print( run_experiments(num_iter, l1_data, l2_data, delta, alpha, beta, lam_seq, gamma_seq, relevance_level = 1) )
print('\n')


alpha, beta = 0.1, 0.2
print(f'alpha: {alpha}, beta: {beta} \n')
print( run_experiments(num_iter, l1_data, l2_data, delta, alpha, beta, lam_seq, gamma_seq, relevance_level = 1) )
print('\n')