from utils import *


# load data
l1_file_name = "Set1_L1.txt"
l2_file_name = "Set1_L2.txt"

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


alpha, beta_seq = 0.1, [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
output = run_experiments_varying_beta(num_iter, l1_data, l2_data, delta, alpha, beta_seq, lam_seq, gamma_seq, relevance_level = 1)
np.save('varying_beta_yahoo.npy', output)