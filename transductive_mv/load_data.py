import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
from utils import compute_kernel_bias, transform_and_normalize
from operator import itemgetter
import pandas as pd
import random
import ipdb

class DataLoader:
    def __init__(self, task_dir, opts):
        self.task_dir = task_dir
        self.n_layer = opts.n_layer
        self.hidden_dim = opts.hidden_dim
        self.constrain_go_depth = opts.constrain_go_depth
        self.n_samp = opts.n_samp

        with open(os.path.join(task_dir, 'Gene_set.txt')) as f:
            self.gene_set = []
            for line in f:
                etype = line.strip()
                self.gene_set.append(etype)

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            self.entitypeid2geneid = dict()
            self.entity2type = dict()
            n_gene = 0
            for line in f:
                entity, eid, etype = line.strip().split()
                self.entity2id[entity] = int(eid)
                self.entity2type[int(eid)] = etype
                if entity in self.gene_set:
                    self.entitypeid2geneid[int(eid)] = n_gene
                    n_gene += 1
        self.n_gene = n_gene
        self.gene_set_id = []
        for g in self.gene_set:
            self.gene_set_id.append(self.entity2id[g])

        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid)

        # ----------------------------------------load textual embedding-------------------------------------- 
        bert_path = '../data/all_entities_pretrain_emb.npy'
        print(bert_path)
        all_entities_pretrain_emb_org = np.load(bert_path)
        # PCA dimension reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.hidden_dim)
        pca.fit(all_entities_pretrain_emb_org)
        all_entities_pretrain_emb = pca.transform(all_entities_pretrain_emb_org)
        self.entity_pretrain_emb = all_entities_pretrain_emb
        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)
        self.filters = defaultdict(lambda:set())

        # ---------------------------------exclude high-level GO terms---------------------------------------#
        # exclude_set1 = ['nucleus', 'cytoplasm', 'membrane', 'protein_binding', 'cytosol', 
        #         'nucleoplasm', 'plasma_membrane', 'integral_component_of_membrane', 
        #         'extracellular_exosome', 'extracellular_region']
        # self.exclude_id1 = itemgetter(*exclude_set1)(self.entity2id)
        self.fact_data = self.read_triples_sl('facts.txt', inv=True)
        self.load_graph(self.fact_data)
        self.load_test_graph(self.fact_data)
        n_rel_org = len(self.relation2id)
        self.n_rel = int(self.KG[:,1].max() + 1)
        if not os.path.exists(os.path.join(self.task_dir, 'train_filtered.txt')):
            print('filtering out triples without answering genes...')
            self.train_data0 = self.read_triples_sl('train.txt', inv=True)
            self.valid_data0 = self.read_triples_sl('valid.txt', inv=True)
            self.test_data0  = self.read_triples_sl('test.txt', inv=True)
            self.train_data = self.delete_triples(self.train_data0, mode='train')
            self.valid_data = self.delete_triples(self.valid_data0, mode='train')
            self.test_data = self.delete_triples(self.test_data0, mode='test')
            
            self.save_filtered_triples(self.task_dir, 'train', self.train_data)
            self.save_filtered_triples(self.task_dir, 'valid', self.valid_data)
            self.save_filtered_triples(self.task_dir, 'test', self.test_data)
            print('# triples filtered out in train: {}'.format(len(self.train_data0) - len(self.train_data)))
            print('# triples filtered out in valid: {}'.format(len(self.valid_data0) - len(self.valid_data)))
            print('# triples filtered out in test: {}'.format(len(self.test_data0) - len(self.test_data)))
        else:
            self.train_data = self.read_triples_sl('train_filtered.txt')#_kgwsl_pathway
            self.valid_data = self.read_triples_sl('valid_filtered.txt')
            self.test_data  = self.read_triples_sl('test_filtered.txt')
        
        self.train_data = np.array(self.train_data) 
        
        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q,  self.test_a  = self.load_query(self.test_data) 

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)

        self.train_q, self.train_a = self.load_query(self.train_data.tolist())
        self.n_train_eval = len(self.train_q)
        self.n_valid_train = len(self.valid_data)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        self.tr_filters = self.get_filter(data='train')    
        self.val_filters = self.get_filter(data='valid')
        self.tst_filters = self.get_filter(data='test')
        self.tr_rm_set_kg = self.get_rm_set(data='train')    
        self.val_rm_set_kg = self.get_rm_set(data='valid')
        self.tst_rm_set_kg = self.get_rm_set(data='test')
        self.tr_rm_set = self.rm_set_kgid2slid(self.tr_rm_set_kg)  
        self.val_rm_set = self.rm_set_kgid2slid(self.val_rm_set_kg)
        self.tst_rm_set = self.rm_set_kgid2slid(self.tst_rm_set_kg)
        config_str = 'n_train(triples):{}, n_valid(answers):{}, n_test(answers):{}'.format(self.n_train, self.n_valid, self.n_test)
        print(config_str)
        with open(opts.perf_file, 'a+') as f:
            f.write(config_str+'\n')

    def read_triples_sl(self, filename, inv=False):
        '''
        filters contain all triples including the triples in the training graph, i.e., all query-answer pairs
        '''
        triples = []
        genes = list(self.entitypeid2geneid.keys())
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                # if h in self.exclude_id1 or t in self.exclude_id1:
                #     continue
                triples.append([h,r,t])
                self.filters[(h,r)].add(t) 
                if inv:
                    if r == self.relation2id['SL_GsG']:
                        triples.append([t,r,h])  
                    elif filename == 'facts.txt':
                        triples.append([t,r+self.n_rel,h])
        return triples

    def get_filter(self, data='valid'):
        filters = defaultdict(lambda: set())
        if data == 'train':
            for triple in self.train_data:
                h, r, t = triple
                filters[(h,r)].add(t)
        elif data == 'valid':
            for triple in self.valid_data:
                h, r, t = triple
                filters[(h,r)].add(t)
        else:
            for triple in self.test_data:
                h, r, t = triple
                filters[(h,r)].add(t)
        return filters
    
    def get_rm_set(self, data='valid'):
        rm_set = defaultdict(lambda: set())
        if data == 'train':
            for k in self.tr_filters.keys():
                if k in self.val_filters.keys():
                    rm_set[k].update(self.val_filters[k])
                if k in self.tst_filters.keys():
                    rm_set[k].update(self.tst_filters[k])
        elif data == 'valid':
            for k in self.val_filters.keys():
                if k in self.tr_filters.keys():
                    rm_set[k].update(self.tr_filters[k])
                if k in self.tst_filters.keys():
                    rm_set[k].update(self.tst_filters[k])
        else:
            for k in self.tst_filters.keys():
                if k in self.tr_filters.keys():
                    rm_set[k].update(self.tr_filters[k])
                if k in self.val_filters.keys():
                    rm_set[k].update(self.val_filters[k])
        return rm_set

    def rm_set_kgid2slid(self, org_dict):
        map_dict = defaultdict(lambda: set())
        for k in org_dict.keys():
            if len(org_dict[k]) != 0:
                val = itemgetter(*org_dict[k])(self.entitypeid2geneid)
                val = np.array([val]).flatten()
                map_dict[k].update(set(val))
        return map_dict    

    def save_filtered_triples(self, directory, filename, triples):
        entity_names = np.array(list(self.entity2id.keys()))
        entity_ids = np.array(list(self.entity2id.values()))

        with open(os.path.join(directory, filename+'_filtered.txt'), 'w') as f:
            for i in range(len(triples)):
                h, r, t = triples[i]
                h_name = entity_names[entity_ids==h][0]
                t_name = entity_names[entity_ids==t][0]
                f.writelines('{} {} {}'.format(h_name, 'SL_GsG',t_name))
                f.write('\n')

    def load_graph(self, triples):
        KG = np.array(triples)
        idd_rel_id = KG[:,1].max() + 1
        # if self.constrain_go_depth == 1:
        #     entity = np.array(tuple(set(self.entity2id.values()) - set(self.exclude_id1)))
        #     n_ent = len(entity)
        entity = np.arange(self.n_ent)
        n_ent = self.n_ent
        idd = np.concatenate([np.expand_dims(entity,1), idd_rel_id*np.ones((n_ent, 1)), np.expand_dims(entity,1)], 1)
        
        self.KG = np.concatenate([KG, idd], 0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_ent))


    def load_test_graph(self, triples):
        KG = np.array(triples)
        idd_rel_id = KG[:,1].max() + 1
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), idd_rel_id*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)
        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:,0])), shape=(self.tn_fact, self.n_ent))

    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers
    
    def get_neighbors(self, nodes, emb, mode='train', last_flag=False):
        if mode=='train':
            KG = self.KG
            M_sub = self.M_sub
        else:
            KG = self.tKG
            M_sub = self.tM_sub
        entity2type = self.entity2type
        def compute_M(data):
            data = np.array(data)
            cols = np.arange(data.size)
            return csr_matrix((cols, (data.ravel(), cols)),
                            shape=(int(data.max() + 1), data.size))

        def get_indices_sparse(data):
            M = compute_M(data)
            return [np.unravel_index(row.data, data.shape) for row in M]

        # nodes: n_node x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_ent, nodes.shape[0]))
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)

        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)
        idd_id = KG[:,1].max()
        idd_edges = np.hstack((nodes, int(idd_id)*np.ones((len(nodes), 1)), nodes[:,[1]]))
        sampled_edges = np.unique(np.vstack((sampled_edges, idd_edges)),axis=0)
        
        if last_flag:
            sampled_edges = sampled_edges[np.isin(sampled_edges[:, -1], list(self.entitypeid2geneid.keys()))]
        
        sampled_edges = torch.LongTensor(sampled_edges).cuda()
        
        # index to nodes, sort by the order of input nodes (i.e., edges[1]), to retrieve the input query order 
        # head nodes are the input query, tail nodes are the corresponding neighbors
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        if last_flag:
            hidden, h0 = emb
            hidden = hidden[np.isin(nodes[:,1], head_nodes[:,1].cpu().numpy()),:]
            h0 = h0[:, np.isin(nodes[:,1], tail_nodes[:,1].cpu().numpy()),:]
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
    
        idd_id = KG[:, 1].max()
        mask = sampled_edges[:,2] == idd_id
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx, (hidden, h0)

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':
            return self.train_data[batch_idx]
    
    def get_batch_sl(self, batch_idx, steps=2, data='train'):
        '''
        return:
        subs: query genes, ids are defined in the whole graph
        rels: query relations
        objs: labeled genes, ids are defined within the range of the number of genes
        '''
        if data=='train':
            query, answer = np.array(self.train_q), np.array(self.train_a)
        if data=='valid':
            query, answer = np.array(self.valid_q), np.array(self.valid_a)
        if data=='test':
            query, answer = np.array(self.test_q), np.array(self.test_a)

        subs = []
        rels = []
        gene_mask = []
        objs = []
        
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objs[i][np.array(answer)[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self,):
        rand_idx = np.random.permutation(self.n_train)
        self.train_data = self.train_data[rand_idx]

    def delete_triples(self, triple, mode='train'):
        triple = np.array(triple)
        num = len(triple) // 200 + (len(triple) % 200 > 0)
        new_triple = []
        for i in range(num): 
            start = i*200
            end = min(len(triple), (i+1)*200)
            batch_idx = np.arange(start, end)
            triple_i = triple[batch_idx,]
            subs, objs = triple_i[:,0], triple_i[:,2]
            nodes = np.stack((np.arange(len(triple_i)), subs), 1)
            for i in range(self.n_layer):
                nodes, _, _, _ = self.get_neighbors(nodes, None, mode=mode)
                nodes = nodes.cpu().numpy()
            tgt_idx = []
            for i in range(len(objs)):
                t = nodes[(nodes[:,0]==i)&(nodes[:,1]==objs[i])]
                if len(t) != 0:
                    tgt_idx.append(i)
            new_triple.append(triple_i[tgt_idx])
        return np.vstack(new_triple).tolist()