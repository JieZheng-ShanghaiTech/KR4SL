import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
from operator import itemgetter
import ipdb

class DataLoader:
    def __init__(self, task_dir, opts):
        self.trans_dir = task_dir
        self.ind_dir = task_dir + '_ind'
        self.n_layer = opts.n_layer
        self.hidden_dim = opts.hidden_dim

        with open(os.path.join(task_dir, 'Gene_set.txt')) as f:
            self.gene_set = []
            for line in f:
                gtype = line.strip()
                self.gene_set.append(gtype)

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            self.entitypeid2geneid = dict() # from gene id in KG to gene id in SL graph
            self.entityid2typeid = dict() # from entity id in KG to entity type id in KG
            self.entity2type = dict()
            n_gene = 0
            for line in f:
                entity, eid, etype = line.strip().split()
                self.entity2id[entity] = int(eid)
                if entity in self.gene_set:
                    self.entitypeid2geneid[int(eid)] = n_gene
                    n_gene += 1                    
        self.n_gene = n_gene
        import ipdb
        
        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            id2relation = []
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid)
                id2relation.append(relation)

        with open(os.path.join(self.ind_dir, 'entities.txt')) as f:
            self.entity2id_ind = dict()
            self.entitypeid2geneid_ind = dict()
            self.entityid2typeid_ind = dict()
            self.entity2type_ind = dict()
            n_gene_ind = 0
            for line in f:
                entity, eid, etype = line.strip().split()
                self.entity2id_ind[entity] = int(eid)
                if entity in self.gene_set:
                    self.entitypeid2geneid_ind[int(eid)] = n_gene_ind
                    n_gene_ind += 1
        self.n_gene_ind = n_gene_ind
        # -----------------load textual embedding-----------#
        all_entities = []
        with open('../data/all_entities.txt') as f:
            for line in f:
                all_entities.append(line)
        all_entities = np.array(all_entities)
        bert_path = '../data/all_entities_pretrain_emb.npy'
        print(bert_path)
        all_entities_pretrain_emb_org = np.load(bert_path)
        # PCA dimension reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.hidden_dim)
        pca.fit(all_entities_pretrain_emb_org)
        all_entities_pretrain_emb = pca.transform(all_entities_pretrain_emb_org)
        sort_idx = all_entities.argsort()
        entity_trans_idx = sort_idx[np.searchsorted(all_entities, np.array([k for k in self.entity2id.keys()]), sorter=sort_idx)]
        entity_ind_idx = sort_idx[np.searchsorted(all_entities, np.array([k for k in self.entity2id_ind.keys()]), sorter=sort_idx)]
        self.entity_pretrain_emb = all_entities_pretrain_emb[entity_trans_idx]
        self.entity_ind_pretrain_emb = all_entities_pretrain_emb[entity_ind_idx]

        for i in range(len(self.relation2id)):
            id2relation.append(id2relation[i] + '_inv')
        id2relation.append('idd')
        self.id2relation = id2relation

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)
        self.n_ent_ind = len(self.entity2id_ind)
        # ---------------------------------exclude high-level GO terms---------------------------------------#
        # exclude_set1 = ['nucleus', 'cytoplasm', 'membrane', 'protein_binding', 'cytosol', 
        #         'nucleoplasm', 'plasma_membrane', 'integral_component_of_membrane', 
        #         'extracellular_exosome', 'extracellular_region']
        # self.exclude_id1 = itemgetter(*exclude_set1)(self.entity2id)
        # self.exclude_id1_ind = itemgetter(*exclude_set1)(self.entity2id_ind)
        
        self.tra_train = self.read_triples_sl(self.trans_dir, 'train.txt', mode='transductive', inv=True)
        self.ind_train = self.read_triples_sl(self.ind_dir, 'train.txt', mode='inductive', inv=True)
        # trans data for training and validation
        self.tra_KG, self.tra_sub = self.load_graph(self.tra_train)
        self.n_rel = int(self.tra_KG[:,1].max() + 1)
        # ind data for test
        self.ind_KG, self.ind_sub = self.load_graph(self.ind_train, mode='inductive')
        
        if not os.path.exists(os.path.join(self.trans_dir, 'valid_filtered.txt')):
            print('filtering out triples without answering genes...')
            self.tra_valid0 = self.read_triples_sl(self.trans_dir, 'valid.txt', mode='transductive', inv=True)
            self.tra_test0  = self.read_triples_sl(self.trans_dir, 'test.txt', mode='transductive', inv=True)
            self.tra_valid = self.delete_triples(self.tra_valid0, mode='transductive')
            self.tra_test = self.delete_triples(self.tra_test0, mode='transductive')
            self.save_filtered_triples(self.trans_dir, 'valid', self.tra_valid, mode='transductive')
            self.save_filtered_triples(self.trans_dir, 'test', self.tra_test, mode='transductive')
            print('# triples filtered out in train: {}'.format(len(self.tra_valid0) - len(self.tra_valid)))
            print('# triples filtered out in valid: {}'.format(len(self.tra_test0) - len(self.tra_test)))
        else:
            self.tra_valid = self.read_triples_sl(self.trans_dir, 'valid_filtered.txt', mode='transductive')
            self.tra_test  = self.read_triples_sl(self.trans_dir, 'test_filtered.txt', mode='transductive')
        if not os.path.exists(os.path.join(self.ind_dir, 'valid_filtered.txt')):
            print('filtering out triples without answering genes...')
            self.ind_valid0 = self.read_triples_sl(self.ind_dir, 'valid.txt', mode='inductive', inv=True)
            self.ind_test0  = self.read_triples_sl(self.ind_dir, 'test.txt',  mode='inductive', inv=True)
            self.ind_valid = self.delete_triples(self.ind_valid0, mode='inductive')
            self.ind_test = self.delete_triples(self.ind_test0, mode='inductive')
            self.save_filtered_triples(self.ind_dir, 'valid', self.ind_valid, mode='inductive')
            self.save_filtered_triples(self.ind_dir, 'test', self.ind_test, mode='inductive')
            print('# triples filtered out in test: {}'.format(len(self.ind_valid0)+len(self.ind_test0) - len(self.ind_valid)-len(self.ind_test)))
        else:
            self.ind_valid = self.read_triples_sl(self.ind_dir, 'valid_filtered.txt', mode='inductive')
            self.ind_test  = self.read_triples_sl(self.ind_dir, 'test_filtered.txt', mode='inductive')
            
        self.val_filters = self.get_filter('valid')
        self.tst_filters = self.get_filter('test')
        
        # all queries and answers, including those in the training graph / inference graphs
        for filt in self.val_filters:
            self.val_filters[filt] = list(self.val_filters[filt])
        for filt in self.tst_filters:
            self.tst_filters[filt] = list(self.tst_filters[filt])
                     
        self.tra_train = np.array(self.tra_valid)
        self.n_train = len(self.tra_train)
        self.tra_val_qry, self.tra_val_ans = self.load_query(self.tra_test)
        self.ind_val_qry, self.ind_val_ans = self.load_query(self.ind_valid)
        self.ind_tst_qry, self.ind_tst_ans = self.load_query(self.ind_test)
        self.valid_q, self.valid_a = self.tra_val_qry, self.tra_val_ans
        self.test_q,  self.test_a  = self.ind_val_qry+self.ind_tst_qry, self.ind_val_ans+self.ind_tst_ans
        
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)
        self.train_q, self.train_a = self.load_query(self.tra_train.tolist())
        self.n_train_eval = len(self.train_q)
        self.n_valid_train = len(self.tra_test)

        config_str = 'n_train(triples):{}, n_valid(answers):{}, n_test(answers):{}'.format(self.n_train, self.n_valid, self.n_test)
        print(config_str)
        with open(opts.perf_file, 'a+') as f:
            f.write(config_str+'\n')

    def read_triples(self, directory, filename, mode='transductive'):
        triples = []
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                if mode == 'transductive':
                    h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                else:
                    h, r, t = self.entity2id_ind[h], self.relation2id[r], self.entity2id_ind[t]

                triples.append([h,r,t])
                triples.append([t, r+self.n_rel, h])
        return triples
    
    def read_triples_sl(self, directory, filename, mode='transductive', inv=False):
        '''
        filters contain all triples including the triples in the training graph, i.e., all query-answer pairs
        '''
        triples = []
        genes = list(self.entitypeid2geneid.keys()) if mode == 'transductive' else list(self.entitypeid2geneid_ind.keys())
        # exclude_id1 = self.exclude_id1 if mode=='transductive' else self.exclude_id1_ind
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                if mode == 'transductive':
                    h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                else:
                    h, r, t = self.entity2id_ind[h], self.relation2id[r], self.entity2id_ind[t]
                # if self.constrain_go_depth == 1:
                #     if h in exclude_id1 or t in exclude_id1:
                #         continue
                triples.append([h,r,t])
                if inv:
                    if r == self.relation2id['SL_GsG']:
                        triples.append([t,r,h])
                    elif filename == 'train.txt':
                        triples.append([t, r+self.n_rel, h])
        return triples
    
    def save_filtered_triples(self, directory, filename, triples, mode='transductive'):
        if mode == 'transductive':
            entity_names = np.array(list(self.entity2id.keys()))
            entity_ids = np.array(list(self.entity2id.values()))
        else:
            entity_names = np.array(list(self.entity2id_ind.keys()))
            entity_ids = np.array(list(self.entity2id_ind.values()))

        with open(os.path.join(directory, filename+'_filtered.txt'), 'w') as f:
            for i in range(len(triples)):
                h, r, t = triples[i]
                h_name = entity_names[entity_ids==h][0]
                t_name = entity_names[entity_ids==t][0]
                f.writelines('{} {} {}'.format(h_name, 'SL_GsG',t_name))
                f.write('\n')

    def load_graph(self, triples, mode='transductive'):
        n_ent_org = self.n_ent if mode=='transductive' else self.n_ent_ind
        entity2id = self.entity2id if mode=='transductive' else self.entity2id_ind
        # exclude_id1 = self.exclude_id1 if mode=='transductive' else self.exclude_id1_ind

        KG = np.array(triples)
        idd_rel_id = KG[:,1].max() + 1
        # if self.constrain_go_depth == 1:
        #     entity = np.array(tuple(set(entity2id.values()) - set(exclude_id1)))
        #     n_ent = len(entity)
        entity = np.arange(n_ent_org)
        n_ent = n_ent_org
        idd = np.concatenate([np.expand_dims(entity,1), idd_rel_id*np.ones((n_ent, 1)), np.expand_dims(entity,1)], 1)
        KG = np.concatenate([KG, idd], 0)
        n_fact = KG.shape[0]        
        M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:,0])), shape=(n_fact, n_ent_org))
        return KG, M_sub

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

    def get_neighbors(self, nodes, emb, mode='transductive', last_flag=False, layer_type=False, layer_node_type=None):
        # nodes: n_node x 2 with (batch_idx, node_idx)

        if mode == 'transductive':
            KG    = self.tra_KG
            M_sub = self.tra_sub
            n_ent = self.n_ent
            gene_idx = list(self.entitypeid2geneid.keys())
            entity2type = self.entity2type
        else:
            KG    = self.ind_KG
            M_sub = self.ind_sub
            n_ent = self.n_ent_ind
            gene_idx = list(self.entitypeid2geneid_ind.keys())
            entity2type = self.entity2type_ind
        import ipdb
        
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)
        
        if last_flag:
            sampled_edges = sampled_edges[np.isin(sampled_edges[:, -1], gene_idx)]
            
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes, tail index: neighbors (tail nodes) 
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)
        
        if last_flag:
            hidden, h0 = emb
            hidden = hidden[np.isin(nodes[:,1], head_nodes[:,1].cpu().numpy()),:]
            h0 = h0[:, np.isin(nodes[:,1], tail_nodes[:,1].cpu().numpy()),:]
            emb = (hidden, h0)

        # get 'idd' relation, i.e., head nodes == tail nodes, representing the query nodes
        idd_id = KG[:, 1].max()
        mask = sampled_edges[:,2] == idd_id
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        return tail_nodes, sampled_edges, old_nodes_new_idx, emb

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':
            return self.tra_train[batch_idx]

    def get_batch_sl(self, batch_idx, steps=2, data='train'):
        if data=='train':
            query, answer = np.array(self.train_q), np.array(self.train_a)
            n_ent = self.n_ent
        if data=='valid':
            query, answer = np.array(self.valid_q), np.array(self.valid_a)
            n_ent = self.n_ent
        if data=='test':
            query, answer = np.array(self.test_q),  np.array(self.test_a)
            n_ent = self.n_ent_ind

        subs = []
        rels = []
        objs = []
        
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self,):
        rand_idx = np.random.permutation(self.n_train)
        self.tra_train = self.tra_train[rand_idx]

    def get_filter(self, data='valid'):
        filters = defaultdict(lambda: set())
        if data == 'valid':
            for triple in self.tra_train:
                h, r, t = triple
                filters[(h,r)].add(t)
            for triple in self.tra_valid:
                h, r, t = triple
                filters[(h,r)].add(t)
            for triple in self.tra_test:
                h, r, t = triple
                filters[(h,r)].add(t)
        else:
            for triple in self.ind_train:
                h, r, t = triple
                filters[(h,r)].add(t)
            for triple in self.ind_valid:
                h, r, t = triple
                filters[(h,r)].add(t)
            for triple in self.ind_test:
                h, r, t = triple
                filters[(h,r)].add(t)
        return filters

    def delete_triples(self, triple, mode='transductive'):
        triple = np.array(triple)     
        num = len(triple) // 100 + (len(triple) % 100 > 0)
        new_triple = []
        for i in range(num): 
            start = i*100
            end = min(len(triple), (i+1)*100)
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