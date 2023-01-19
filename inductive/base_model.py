import torch
import numpy as np
import time
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from models import RED_GNN_induc
from utils import cal_ranks, cal_performance, cal_ndcg
from operator import itemgetter
import ipdb

class BaseModel(object):
    def __init__(self, args, loader):
        self.model = RED_GNN_induc(args, loader)
        self.model.cuda()

        self.perf_file = args.perf_file

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_ent_ind = loader.n_ent_ind
        self.n_batch = args.n_batch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer

        self.n_train_eval = loader.n_train_eval
        self.n_valid_train = loader.n_valid_train 

        self.n_gene = loader.n_gene
        self.gene_idx = np.array(list(loader.entitypeid2geneid.keys()))
        self.n_gene_ind = loader.n_gene_ind
        self.gene_idx_ind = np.array(list(loader.entitypeid2geneid_ind.keys()))

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0
        self.seed = args.seed
        
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        

    def train_batch(self,):
        epoch_loss = 0
        i = 0
        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)
        
        t_time = time.time()
        self.model.train()
        for i in range(n_batch):
            start = i*batch_size
            end = min(self.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            pos_triple = self.loader.get_batch(batch_idx, data='train')

            self.model.zero_grad()
            pos_scores, _, _ = self.model(pos_triple[:,0], pos_triple[:,1])
            loss = self.model.loss(pos_scores, pos_triple[:,2])
            loss.backward()
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
        self.t_time += time.time() - t_time
        config_str = 'Training loss: {}'.format(epoch_loss/self.n_train)
        print(config_str)
        with open(self.perf_file, 'a+') as f:
            f.write(config_str+'\n')
        self.scheduler.step() 
        self.loader.shuffle_train()

    def evaluate(self, data='train'):
        batch_size = self.n_batch
        if data == 'train':
            n_data = self.n_train_eval
            mode = 'transductive'
            n_gene = self.n_gene
            filters_obj = self.loader.val_filters
            filters_obj_gene = self.loader.entitypeid2geneid
        elif data == 'valid':
            n_data = self.n_valid
            mode = 'transductive'
            n_gene = self.n_gene
            filters_obj = self.loader.val_filters
            filters_obj_gene = self.loader.entitypeid2geneid
        elif data == 'test':
            n_data = self.n_test
            mode = 'inductive'
            n_gene = self.n_gene_ind
            filters_obj = self.loader.tst_filters
            filters_obj_gene = self.loader.entitypeid2geneid_ind

        ranking = []
        s_all, o_all, f_all = [], [], []

        batch_size = 50
        self.model.eval()
        with torch.no_grad():
            n_batch = n_data // batch_size + (n_data % batch_size > 0)
            for i in range(n_batch):
                start = i*batch_size
                end = min(n_data, (i+1)*batch_size)
                batch_idx = np.arange(start, end)
                
                subs, rels, objs = self.loader.get_batch_sl(batch_idx, data=data)
                scores, alpha_all, edges_all = self.model(subs, rels, mode=mode, train_mode='eval')
                scores = scores.data.cpu().numpy()

                gene_idx = np.array(list(filters_obj_gene.keys()))
                scores = scores[:, gene_idx]
                objs = objs[:, gene_idx]                        
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks
                s_all += scores.tolist()
                o_all += objs.tolist()
                f_all += filters.tolist()

            ranking = np.array(ranking)
            mrr, h1, h10 = cal_performance(ranking)
            s_all = np.vstack(s_all)
            o_all = np.vstack(o_all)
            f_all = np.vstack(f_all)
            ndcg_10, p_10, r_10 = cal_ndcg(s_all, o_all, f_all, n=10)
            ndcg_20, p_20, r_20 = cal_ndcg(s_all, o_all, f_all, n=20)
            ndcg_50, p_50, r_50 = cal_ndcg(s_all, o_all, f_all, n=50)
            out_str = '[%s] MRR:%.4f H@1:%.4f H@10:%.4f \nNDCG@10:%.4f NDCG@20:%.4f NDCG@50:%.4f\nP@10:%.4f P@20:%.4f P@50:%.4f\nR@10:%.4f R@20:%.4f R@50:%.4f\n\n' %(data.upper(), mrr, h1, h10, ndcg_10, ndcg_20, ndcg_50, p_10, p_20, p_50, r_10, r_20, r_50)
        return ndcg_50, out_str, (mrr, h1, h10, ndcg_10, ndcg_20, ndcg_50, p_10, p_20, p_50, r_10, r_20, r_50)
