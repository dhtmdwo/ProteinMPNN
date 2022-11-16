import argparse
import os.path

def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor    
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN

    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint
    #==========feature===================
    logfile = base_folder + 'log.txt'
    logfile1 = base_folder + 'log1.txt'
    logfile3 = base_folder + 'log3.txt'
    logfile6 = base_folder + 'log6.txt'
    logfile8 = base_folder + 'log8.txt'
    #==========enc0===================
    logfile15 = base_folder + 'log15.txt'
    logfile17 = base_folder + 'log17.txt'
    logfile19 = base_folder + 'log19.txt'
    logfile21 = base_folder + 'log21.txt'
    logfile23 = base_folder + 'log23.txt'
    logfile25 = base_folder + 'log25.txt'
    logfile27 = base_folder + 'log27.txt'
    logfile29 = base_folder + 'log29.txt'
    #==========enc1===================
    logfile37 = base_folder + 'log37.txt'
    logfile39 = base_folder + 'log39.txt'
    logfile41 = base_folder + 'log41.txt'
    logfile43 = base_folder + 'log43.txt'
    logfile45 = base_folder + 'log45.txt'
    logfile47 = base_folder + 'log47.txt'
    logfile49 = base_folder + 'log49.txt'
    logfile51 = base_folder + 'log51.txt'
    #==========enc2===================
    logfile59 = base_folder + 'log59.txt'
    logfile61 = base_folder + 'log61.txt'
    logfile63 = base_folder + 'log63.txt'
    logfile65 = base_folder + 'log65.txt'
    logfile67 = base_folder + 'log67.txt'
    logfile69 = base_folder + 'log69.txt'
    logfile71 = base_folder + 'log71.txt'
    logfile73 = base_folder + 'log73.txt'
    #==========dec0===================
    logfile79 = base_folder + 'log79.txt'
    logfile81 = base_folder + 'log81.txt'
    logfile83 = base_folder + 'log83.txt'
    logfile85 = base_folder + 'log85.txt'
    logfile87 = base_folder + 'log87.txt'
    #==========dec1===================
    logfile93 = base_folder + 'log93.txt'
    logfile95 = base_folder + 'log95.txt'
    logfile97 = base_folder + 'log97.txt'
    logfile99 = base_folder + 'log99.txt'
    logfile101 = base_folder + 'log101.txt'
    #==========dec2===================
    logfile107 = base_folder + 'log107.txt'
    logfile109 = base_folder + 'log109.txt'
    logfile111 = base_folder + 'log111.txt'
    logfile113 = base_folder + 'log113.txt'
    logfile115 = base_folder + 'log115.txt'

    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')
        
        with open(logfile1, 'w') as f1:
            f1.write('Epoch\tTrain\tValidation\n')
        with open(logfile3, 'w') as f3:
            f3.write('Epoch\tTrain\tValidation\n')
        with open(logfile6, 'w') as f6:
            f6.write('Epoch\tTrain\tValidation\n')
        with open(logfile8, 'w') as f8:
            f8.write('Epoch\tTrain\tValidation\n')

        with open(logfile15, 'w') as f15:
            f15.write('Epoch\tTrain\tValidation\n')
        with open(logfile17, 'w') as f17:
            f17.write('Epoch\tTrain\tValidation\n')
        with open(logfile19, 'w') as f19:
            f19.write('Epoch\tTrain\tValidation\n')
        with open(logfile21, 'w') as f21:
            f21.write('Epoch\tTrain\tValidation\n')
        with open(logfile23, 'w') as f23:
            f23.write('Epoch\tTrain\tValidation\n')
        with open(logfile25, 'w') as f25:
            f25.write('Epoch\tTrain\tValidation\n')
        with open(logfile27, 'w') as f27:
            f27.write('Epoch\tTrain\tValidation\n')
        with open(logfile29, 'w') as f29:
            f29.write('Epoch\tTrain\tValidation\n')

        with open(logfile37, 'w') as f37:
            f37.write('Epoch\tTrain\tValidation\n')
        with open(logfile39, 'w') as f39:
            f39.write('Epoch\tTrain\tValidation\n')
        with open(logfile41, 'w') as f41:
            f41.write('Epoch\tTrain\tValidation\n')
        with open(logfile43, 'w') as f43:
            f43.write('Epoch\tTrain\tValidation\n')
        with open(logfile45, 'w') as f45:
            f45.write('Epoch\tTrain\tValidation\n')
        with open(logfile47, 'w') as f47:
            f47.write('Epoch\tTrain\tValidation\n')
        with open(logfile49, 'w') as f49:
            f49.write('Epoch\tTrain\tValidation\n')
        with open(logfile51, 'w') as f51:
            f51.write('Epoch\tTrain\tValidation\n')

        with open(logfile59, 'w') as f59:
            f59.write('Epoch\tTrain\tValidation\n')
        with open(logfile61, 'w') as f61:
            f61.write('Epoch\tTrain\tValidation\n')
        with open(logfile63, 'w') as f63:
            f63.write('Epoch\tTrain\tValidation\n')
        with open(logfile65, 'w') as f65:
            f65.write('Epoch\tTrain\tValidation\n')
        with open(logfile67, 'w') as f67:
            f67.write('Epoch\tTrain\tValidation\n')
        with open(logfile69, 'w') as f69:
            f69.write('Epoch\tTrain\tValidation\n')
        with open(logfile71, 'w') as f71:
            f71.write('Epoch\tTrain\tValidation\n')
        with open(logfile73, 'w') as f73:
            f73.write('Epoch\tTrain\tValidation\n')

        with open(logfile79, 'w') as f79:
            f79.write('Epoch\tTrain\tValidation\n')
        with open(logfile81, 'w') as f81:
            f81.write('Epoch\tTrain\tValidation\n')
        with open(logfile83, 'w') as f83:
            f83.write('Epoch\tTrain\tValidation\n')
        with open(logfile85, 'w') as f85:
            f85.write('Epoch\tTrain\tValidation\n')
        with open(logfile87, 'w') as f87:
            f87.write('Epoch\tTrain\tValidation\n')

        with open(logfile93, 'w') as f93:
            f93.write('Epoch\tTrain\tValidation\n')
        with open(logfile95, 'w') as f95:
            f95.write('Epoch\tTrain\tValidation\n')
        with open(logfile97, 'w') as f97:
            f97.write('Epoch\tTrain\tValidation\n')
        with open(logfile99, 'w') as f99:
            f99.write('Epoch\tTrain\tValidation\n')
        with open(logfile101, 'w') as f101:
            f101.write('Epoch\tTrain\tValidation\n')

        with open(logfile107, 'w') as f107:
            f107.write('Epoch\tTrain\tValidation\n')
        with open(logfile109, 'w') as f109:
            f109.write('Epoch\tTrain\tValidation\n')
        with open(logfile111, 'w') as f111:
            f111.write('Epoch\tTrain\tValidation\n')
        with open(logfile113, 'w') as f113:
            f113.write('Epoch\tTrain\tValidation\n')
        with open(logfile115, 'w') as f115:
            f115.write('Epoch\tTrain\tValidation\n')



    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 4}

   
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)
     
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)


    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)


    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)


    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        for i in range(3):
            q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()
       
        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        
        reload_c = 0 
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                reload_c += 1
            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
           
                    scaler.scale(loss_av_smoothed).backward()
                     
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()
                
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                for _, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            
            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
                f.write('\n')
            
            number = 1
            for name, params in model.named_parameters():
              if number ==1:
                weights = str(model.get_parameter(name))
                with open(logfile1, 'a') as f1:
                  f1.write(weights+'\n')
              if number ==3:
                weights = str(model.get_parameter(name))
                with open(logfile3, 'a') as f3:
                  f3.write(weights+'\n')
              if number ==6:
                weights = str(model.get_parameter(name))
                with open(logfile6, 'a') as f6:
                  f6.write(weights+'\n')
              if number ==8:
                weights = str(model.get_parameter(name))
                with open(logfile8, 'a') as f8:
                  f8.write(weights+'\n')

              if number ==15:
                weights = str(model.get_parameter(name))
                with open(logfile15, 'a') as f15:
                  f15.write(weights+'\n')
              if number ==17:
                weights = str(model.get_parameter(name))
                with open(logfile17, 'a') as f17:
                  f17.write(weights+'\n')
              if number ==19:
                weights = str(model.get_parameter(name))
                with open(logfile19, 'a') as f19:
                  f19.write(weights+'\n')
              if number ==21:
                weights = str(model.get_parameter(name))
                with open(logfile21, 'a') as f21:
                  f21.write(weights+'\n')
              if number ==23:
                weights = str(model.get_parameter(name))
                with open(logfile23, 'a') as f23:
                  f23.write(weights+'\n')
              if number ==25:
                weights = str(model.get_parameter(name))
                with open(logfile25, 'a') as f25:
                  f25.write(weights+'\n')
              if number ==27:
                weights = str(model.get_parameter(name))
                with open(logfile27, 'a') as f27:
                  f27.write(weights+'\n')
              if number ==29:
                weights = str(model.get_parameter(name))
                with open(logfile29, 'a') as f29:
                  f29.write(weights+'\n')

              if number ==37:
                weights = str(model.get_parameter(name))
                with open(logfile37, 'a') as f37:
                  f37.write(weights+'\n')
              if number ==39:
                weights = str(model.get_parameter(name))
                with open(logfile39, 'a') as f39:
                  f39.write(weights+'\n')
              if number ==41:
                weights = str(model.get_parameter(name))
                with open(logfile41, 'a') as f41:
                  f41.write(weights+'\n')
              if number ==43:
                weights = str(model.get_parameter(name))
                with open(logfile43, 'a') as f43:
                  f43.write(weights+'\n')
              if number ==45:
                weights = str(model.get_parameter(name))
                with open(logfile45, 'a') as f45:
                  f45.write(weights+'\n')
              if number ==47:
                weights = str(model.get_parameter(name))
                with open(logfile47, 'a') as f47:
                  f47.write(weights+'\n')
              if number ==49:
                weights = str(model.get_parameter(name))
                with open(logfile49, 'a') as f49:
                  f49.write(weights+'\n')
              if number ==51:
                weights = str(model.get_parameter(name))
                with open(logfile51, 'a') as f51:
                  f51.write(weights+'\n')

              if number ==59:
                weights = str(model.get_parameter(name))
                with open(logfile59, 'a') as f59:
                  f59.write(weights+'\n')
              if number ==61:
                weights = str(model.get_parameter(name))
                with open(logfile61, 'a') as f61:
                  f61.write(weights+'\n')
              if number ==63:
                weights = str(model.get_parameter(name))
                with open(logfile63, 'a') as f63:
                  f63.write(weights+'\n')
              if number ==65:
                weights = str(model.get_parameter(name))
                with open(logfile65, 'a') as f65:
                  f65.write(weights+'\n')
              if number ==67:
                weights = str(model.get_parameter(name))
                with open(logfile67, 'a') as f67:
                  f67.write(weights+'\n')
              if number ==69:
                weights = str(model.get_parameter(name))
                with open(logfile69, 'a') as f69:
                  f69.write(weights+'\n')
              if number ==71:
                weights = str(model.get_parameter(name))
                with open(logfile71, 'a') as f71:
                  f71.write(weights+'\n')
              if number ==73:
                weights = str(model.get_parameter(name))
                with open(logfile73, 'a') as f73:
                  f73.write(weights+'\n')

              if number ==79:
                weights = str(model.get_parameter(name))
                with open(logfile79, 'a') as f79:
                  f79.write(weights+'\n')
              if number ==81:
                weights = str(model.get_parameter(name))
                with open(logfile81, 'a') as f81:
                  f81.write(weights+'\n')
              if number ==83:
                weights = str(model.get_parameter(name))
                with open(logfile83, 'a') as f83:
                  f83.write(weights+'\n')
              if number ==85:
                weights = str(model.get_parameter(name))
                with open(logfile85, 'a') as f85:
                  f85.write(weights+'\n')
              if number ==87:
                weights = str(model.get_parameter(name))
                with open(logfile87, 'a') as f87:
                  f87.write(weights+'\n')

              if number ==93:
                weights = str(model.get_parameter(name))
                with open(logfile93, 'a') as f93:
                  f93.write(weights+'\n')
              if number ==95:
                weights = str(model.get_parameter(name))
                with open(logfile95, 'a') as f95:
                  f95.write(weights+'\n')
              if number ==97:
                weights = str(model.get_parameter(name))
                with open(logfile97, 'a') as f97:
                  f97.write(weights+'\n')
              if number ==99:
                weights = str(model.get_parameter(name))
                with open(logfile99, 'a') as f99:
                  f99.write(weights+'\n')
              if number ==101:
                weights = str(model.get_parameter(name))
                with open(logfile101, 'a') as f101:
                  f101.write(weights+'\n')

              if number ==107:
                weights = str(model.get_parameter(name))
                with open(logfile107, 'a') as f107:
                  f107.write(weights+'\n')
              if number ==109:
                weights = str(model.get_parameter(name))
                with open(logfile109, 'a') as f109:
                  f109.write(weights+'\n')
              if number ==111:
                weights = str(model.get_parameter(name))
                with open(logfile111, 'a') as f111:
                  f111.write(weights+'\n')
              if number ==113:
                weights = str(model.get_parameter(name))
                with open(logfile113, 'a') as f113:
                  f113.write(weights+'\n')
              if number ==115:
                weights = str(model.get_parameter(name))
                with open(logfile115, 'a') as f115:
                  f115.write(weights+'\n')


            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
            print("train weight : "+str(train_weights)+'\n')
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
 
    args = argparser.parse_args()    
    main(args)   
