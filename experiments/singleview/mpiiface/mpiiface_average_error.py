import os
import numpy as np

MODEL_DIR =  'vertex/MPII/test_0'
LOG_DIR = 'log'
log_dir_base = f'{LOG_DIR}/{MODEL_DIR}'

losses_subs = {}
log_dirs = os.listdir(log_dir_base)
log_dirs.sort(key = lambda p: int(p.split('_')[-1]))
for log_dir in log_dirs:
    sub = log_dir.split('_')[-1]
    # read file
    log_file = f'{log_dir_base}/{log_dir}/test_log.txt'
    with open(log_file, 'r') as f:
        lines = f.readlines()
    # get best loss
    losses_epochs = []
    for line in lines:
        losses_epochs += [float(line.split(',')[6].split(' ')[2])]
    loss_best = min(losses_epochs)
    # track loss
    losses_subs[sub] = loss_best

# avg loss
loss_avg = np.mean([loss for loss in losses_subs.values()])

print(f'n subjects: {len(losses_subs)}')
print(f'loss per subject: {losses_subs}')
print(f'average loss: {loss_avg}')
    
    



