import numpy as np
import sys

from arguments import args_parser
parser = args_parser()
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
import datetime
import random

from torch.utils.data import ConcatDataset, Dataset, DataLoader
from utils import Discretizer, Normalizer, my_metrics, is_ascending, robust_metrics
from dataset.dataloader_MIRACL import get_multimodal_datasets
from mymodel.MIRACL_patient import MIRACL
from mimic4noisy.MIRACL_utils import select_memorization_and_forgetting_per_epoch, fit_gaussian_model_loss_corr_latest, my_collate, read_timeseries
from mimic4noisy.add_noisy_label import flip_label, flip_multilabel
from mimic4noisy.gaussian_model import fit_gaussian_model
import time


# Print current argument setting:
# Print args to display settings at the start of the program
print("Program started with the following arguments:")
print(args)



torch.autograd.set_detect_anomaly(True)

num_workers = args.num_workers
adjust_step = 2

if args.device != "cpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
args.task = args.task.split(',')



task_weight = {'in-hospital-mortality':0.2,
               'length-of-stay':0.5,
                'phenotyping':1,
               'decompensation':0.2,
                'readmission':0.2,
                'diagnosis':0.2}


def main(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Running with seed: {seed}")
    
    
    if args.corr:
        print("Correction Module Will Start in the Training")
        

    # Read Data and Preprocessing
    # Time series data discretization
    discretizer = Discretizer(timestep=float(args.timestep), store_masks=True, impute_strategy='previous', start_time='zero')
    discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    # Mutli-task dataloader
    mutli_train_dl = []
    mutli_test_dl = []
    num_train_samples = []

    
    for t in range(len(args.task)):
        normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        normalizer_state = args.normalizer_state
        if normalizer_state is None:
            normalizer_state = 'normalizers/ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(1.0)
            normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        normalizer.load_params(normalizer_state)

        train_ds, test_ds = get_multimodal_datasets(discretizer, normalizer, args, args.task[t])
        
         
        # Add noisy target according to args noise ratios and args type
        target = train_ds.y
        if args.task[t] in ['phenotyping','diagnosis']:
            
            noisy_target, mask = flip_multilabel(t, target, args.noise_ratio, args.noise_type, args)
            

        else:
            noisy_target, mask = flip_label(t, target, args.noise_ratio, args.noise_type, args)
        train_ds.set_noisy_labels(noisy_target)
        
        train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=num_workers, drop_last=False)
        test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=num_workers, drop_last=False)
        
        mutli_train_dl.append(train_dl)
        mutli_test_dl.append(test_dl)

        num_train_samples.append(len(train_ds))
    

    # Fit Models
    model = MIRACL(hidden_dim=args.hidden_dim, layers=4, expert_k=2, expert_total=10, device=device).to(device)
    criterion = torch.nn.BCELoss()
    each_criterion = torch.nn.BCELoss(reduction='none') 
    criterion_ce = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    multi_lr = [args.lr for i in range(len(args.task))]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    print(f'Current Noise Setting: {args.noise_type}, {args.noise_ratio}')

    file_path = 'log/['+args.model+']' + '_seed_' + str(seed) + '_epoch_' + str(args.epochs) + '_' + str(args.noise_type) + '_' + str(args.noise_ratio) + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
    
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    if not os.path.exists('log'):
        os.mkdir('log')
    start_time = time.time()
        
        
    # Build Loss Dictionary
    loss_all = {}
    pred_all = {}
    ranks_all = {}
    true_all = {}
    noisy_all = {}
    mems_all = {}
    cons_all = {}
    probs_all = {}
    note_all = {}
    ehr_all = {}
    corrs_all = {}
    print(num_train_samples[t])
    for t in range(len(mutli_train_dl)):
        # Multi-Label Setting
        if args.task[t] in ['phenotyping', 'diagnosis']:
            loss_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            true_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            pred_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            noisy_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            ranks_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            mems_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            cons_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            probs_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            ehr_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            
            note_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            corrs_all[t] = np.zeros((num_train_samples[t], args.nbins[t], args.epochs))
            
    


    # Starts Training
    for epoch in tqdm(range(1, args.epochs+1)):
        print('Epoch:', epoch)
        # Train
        model.train()
        train_loss = 0
        task_list = list(range(len(mutli_train_dl)))
        for t in range(len(mutli_train_dl)):
            task_now = args.task[t]
            print('Task:', task_now, ' Training!')

            
            if len(args.task) > 1:
                optimizer.param_groups[0]['lr'] = multi_lr[t]
                
            
            with tqdm(mutli_train_dl[t], position=0, ncols=150, colour='#666666') as tqdm_range:
                for i, data in enumerate(tqdm_range):
                    optimizer.zero_grad()
                    ehr, ehr_length, mask_ehr, note, mask_note, label, noisy_label, task_index, x_ids, patient_ids = data
                    
                    
                    ehr = torch.from_numpy(ehr).float().to(device)
                    mask_ehr = torch.from_numpy(mask_ehr).long().to(device)
                    mask_note = torch.from_numpy(mask_note).long().to(device)
                    y_true = torch.from_numpy(label).float().to(device)
                    task_index = torch.from_numpy(task_index).long().to(device)
                    y_noisy = torch.from_numpy(noisy_label).float().to(device)
                    patient_ids = torch.from_numpy(patient_ids).float().to(device)
                    
                    y_pred = model(ehr, ehr_length, mask_ehr, note, mask_note, task_index, patient_ids)
                    
                    criterion_now = criterion
                    criterion_now_each = each_criterion

                    y_pred, ortho_loss, moe_loss, ehr_scores, note_scores, L_align = y_pred
                             

                    if task_now == 'diagnosis':
                        # Assume Negative Label for missing label
                        y_noisy[y_noisy <= -1] = 0

                    loss = criterion_now(y_pred, y_noisy)
                    each_loss = criterion_now_each(y_pred, y_noisy)
                    
                    
                    
                    
                    ehr_loss = criterion(ehr_scores, y_noisy)
                    note_loss = criterion(note_scores, y_noisy)
                    
                    if task_now in ['phenotyping','diagnosis']:
                        y_pred_one_hot = (y_pred > 0.5).int()
                        
                        # Compute ranks within each instance
                        _, sorted_indices = torch.sort(y_pred, dim=1, descending=True)  # Sort in descending order
                        ranks = torch.empty_like(sorted_indices)  # Placeholder for ranks


                        # Assign ranks (1 for the highest probability)
                        for i in range(y_pred.size(0)):  # Iterate over each instance
                            ranks[i, sorted_indices[i]] = torch.arange(1, y_pred.size(1) + 1, device=y_pred.device)
 

                        
                        loss_all[t][x_ids,:, epoch-1] = each_loss.data.detach().clone().cpu().numpy()
                        pred_all[t][x_ids,:, epoch-1] = y_pred_one_hot.data.detach().clone().cpu().numpy()
                        ranks_all[t][x_ids,:, epoch-1] = ranks.data.detach().clone().cpu().numpy()
                        true_all[t][x_ids,:, epoch-1] = y_true.data.detach().clone().cpu().numpy()
                        noisy_all[t][x_ids,:, epoch-1] = y_noisy.data.detach().clone().cpu().numpy()
                        probs_all[t][x_ids,:, epoch-1] = y_pred.data.detach().clone().cpu().numpy()
                        ehr_all[t][x_ids,:, epoch-1] = ehr_scores.data.detach().clone().cpu().numpy()
                        note_all[t][x_ids,:, epoch-1] = note_scores.data.detach().clone().cpu().numpy()   
                        

                   # Correction Starts
                    if epoch > args.warmup and args.corr:
                        
                        # # Add weight decay to prevent overfit to correction label
                        optimizer.param_groups[0]['weight_decay'] = 1e-5
                        
                        initial_coeff = 1 # Starting value
                        final_coeff = 0.5   # Final value
                        total_epochs = args.epochs + 1   # Total training epochs

                        # Dynamically adjust coefficient based on the epoch
                        
                        dynamic_coeff = initial_coeff + (final_coeff - initial_coeff) * (epoch / total_epochs)
                        corr_loss = criterion_now_each(y_pred, corr_labels[x_ids, :].to(torch.float32))
                        
                        # Add BCE Loss
                        weight_bce_loss = dynamic_coeff * corr_loss + (1-dynamic_coeff) * each_loss
                        
                        total_loss = weight_bce_loss.mean()
                        
                        
                        # Final loss with regularization applied outside corr_loss
                        loss = (total_loss + ortho_loss * 0.5 + moe_loss +  args.cont_coef * (ehr_loss + note_loss + L_align)) * task_weight[task_now]
                    else: 
                        loss = (loss + ortho_loss * 0.5 + moe_loss +  args.cont_coef * (ehr_loss + note_loss + L_align))*task_weight[task_now]
                    

                    train_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()
                    
            scheduler.step()
                
            # scheduler.step(train_loss)
        print(f'Current train loss: {train_loss}')
            
        
        # Calculate selection metric fit for warmup -1 epoch
        if epoch > args.warmup-1:
            
            Mem, Forg, C = select_memorization_and_forgetting_per_epoch(pred_all[t], noisy_all[t][:,:,epoch-1], true_all[t][:,:,epoch-1], epoch-1)
            mems_all[t][:,:, epoch-1] = C.data.detach().clone().cpu().numpy()
            
            
          
            corr_loss, corr_rank_loss, clean_mask, negative_pair_mask, corr_labels = fit_gaussian_model_loss_corr_latest(
                args,   
                model,
                torch.tensor(loss_all[t][:, :, epoch-1], device=device),  # Directly use epoch-1 slice
                torch.tensor(ranks_all[t][:, :, epoch-1], device=device),  
                C, 
                torch.tensor(cons_all[t][:, :, epoch-1], device=device),
                torch.tensor(ehr_all[t][:, :, epoch-1], device=device),
                torch.tensor(note_all[t][:, :, epoch-1], device=device),
                criterion_now_each, 
                torch.tensor(noisy_all[t][:, :, epoch-1], device=device).clone(),  # Convert and clone
                torch.tensor(probs_all[t][:, :, epoch-1], device=device), 
                torch.tensor(true_all[t][:, :, epoch-1], device=device),
                epoch-1
            )
            
            corrs_all[t][:,:, epoch-1] = corr_labels.data.detach().clone().cpu().numpy()

            
            
        else:
            corr_labels = torch.tensor(noisy_all[t][:,:,epoch-1], device=device).clone()
        
        
        
        # Test
        with torch.no_grad():
            model.eval()
            test_loss = 0
            test_auc = []
            test_aupr = []
            test_mAP = []
            test_f1 = []
            test_f1_class = []
            for t in range(len(mutli_test_dl)):
                task_now = args.task[t]
                with tqdm(mutli_test_dl[t], position=0, ncols=150, colour='#666666') as tqdm_range:
                    outGT = torch.FloatTensor().to(device)
                    outPRED = torch.FloatTensor().to(device)
                    for i, data in enumerate(tqdm_range):
                        ehr, ehr_length, mask_ehr, note, mask_note, label, noisy_label, task_index, x_ids, patient_ids = data
                        ehr = torch.from_numpy(ehr).float().to(device)
                        mask_ehr = torch.from_numpy(mask_ehr).long().to(device)
                        mask_note = torch.from_numpy(mask_note).long().to(device)
                        y_true = torch.from_numpy(label).float().to(device)
                        task_index = torch.from_numpy(task_index).long().to(device)
                        patient_ids = torch.from_numpy(patient_ids).float().to(device)

                        y_pred = model(ehr, ehr_length, mask_ehr, note, mask_note, task_index, patient_ids)

                        
                        # Choose loss function based on the task
                        if task_now in ['length-of-stay','drg']:
                            criterion_now = criterion_ce
                            y_true = y_true.long().view(-1)
                        else:
                            criterion_now = criterion

                        y_pred = y_pred.reshape(ehr.shape[0], -1)
                        if task_now == 'diagnosis':
                            # Assume Negative Label for missing label
                            y_true[y_true <= -1] = 0
                            loss = criterion_now(y_pred, y_true)
                        else:
                            loss = criterion_now(y_pred, y_true)

                        test_loss += loss.item()

                        if task_now in ['length-of-stay','drg']:
                            _, y_pred = torch.max(y_pred, dim=1)

                        outPRED = torch.cat((outPRED, y_pred), 0)
                        outGT = torch.cat((outGT, y_true), 0)
                
                

                auc, aupr = my_metrics(outGT, outPRED, task_now)
                mAP, f1, f1_class = robust_metrics(outGT, outPRED, task_now)
                test_mAP.append(mAP)
                test_f1.append(f1)
                test_f1_class.append(f1_class)
                test_auc.append(auc)
                test_aupr.append(aupr)
                print('Task: ', task_now, ' Test Macro AUC:', auc, '   AUPR:', aupr)
                print('Task: ', task_now, ' Test mAP:', mAP, '   f1:', f1, ' Test f1 class:', f1_class)

                with open(file_path, "a", encoding='utf-8') as f:
                    f.write(f'Task: {task_now} Test mAP: {mAP}   f1: {f1} Test f1 class: {f1_class}\n')


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model runtime: {elapsed_time:.4f} seconds")
    return test_mAP[-1], test_f1[-1], test_f1_class[-1]
        

               
if __name__ == '__main__':
    seeds = args.seed
    test_map, test_f1, test_f1_class = [], [], []
    
    for seed in seeds:
        # Call the main function and collect the statistics
        map_val, f1_val, f1_class_val = main(seed)
        test_map.append(map_val)
        test_f1.append(f1_val)
        test_f1_class.append(f1_class_val)
    

    # Calculate mean statistics
    mean_map = np.mean(test_map)
    std_map = np.std(test_map)

    mean_f1 = np.mean(test_f1)
    std_f1 = np.std(test_f1)

    mean_f1_class = np.mean(test_f1_class)
    std_f1_class = np.std(test_f1_class)

    # Print the results
    print(f"Mean Test mAP: {mean_map:.4f} ± {std_map:.4f}")
    print(f"Mean Test F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Mean Test F1 Class: {mean_f1_class:.4f} ± {std_f1_class:.4f}")
