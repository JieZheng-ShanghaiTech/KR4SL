import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter
import ipdb

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(n_rel, in_dim)
        if combine_by == 'sum':
            self.W1 = nn.Linear(in_dim, in_dim, bias=False)
        elif combine_by == 'concat':
            self.W1 = nn.Linear(3*in_dim, in_dim, bias=False)
        elif combine_by == 'att' or 'worel':
            self.W1 = nn.Linear(in_dim, in_dim, bias=False)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)        
        self.w_alpha  = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)        

    def forward(self, q_sub, hidden, edges, n_node, old_nodes_new_idx, entity_pretrain_emb, h_sub):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]
        
        hs = hidden[sub]
        hr = self.rela_embed(rel)  #emb of all neighboring edges
        message = self.W1(hs + hr + h_sub)
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(message))))
        
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new, alpha

class RED_GNN_trans(torch.nn.Module):
    def __init__(self, params, loader):
        super(RED_GNN_trans, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]
        self.gnn_layers = []
         
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)         # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        for m in self.gnn_layers.modules():
            self.weights_init(m)
        self.weights_init(self.W_final)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        if isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight.data)

    def forward(self, subs, rels, mode='train', explain=False, train_mode='train'):
        n = len(subs)
        n_gene = self.loader.n_gene
        self.gene_idx = list(self.loader.entitypeid2geneid.keys())
        entity_pretrain_emb = self.loader.entity_pretrain_emb

        q_sub = torch.LongTensor(subs).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)

        h0 = torch.zeros((1, n,self.hidden_dim)).cuda()
        hidden = torch.zeros(n, self.hidden_dim).cuda()
        entity_pretrain_emb = torch.tensor(entity_pretrain_emb, dtype=torch.float32)
        hidden = entity_pretrain_emb[subs].cuda().view(n, self.hidden_dim)

        alpha_all = []
        edges_all = []
        for i in range(self.n_layer):
            if i != self.n_layer-1:
                nodes, edges, old_nodes_new_idx, _ = self.loader.get_neighbors(nodes.data.cpu().numpy(), None, mode=mode, last_flag=False)#nodes: neighbors of source nodes   
            else:
                nodes, edges, old_nodes_new_idx, (hidden,h0) = self.loader.get_neighbors(nodes.data.cpu().numpy(), (hidden,h0), mode=mode, last_flag=True)#nodes: neighbors of source nodes   
                h_sub = entity_pretrain_emb[edges[:,1]].cuda() + entity_pretrain_emb[edges[:,3]].cuda()

            hidden, alpha = self.gnn_layers[i](q_sub, hidden, edges, nodes.size(0), old_nodes_new_idx, entity_pretrain_emb, h_sub)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
                
            # hidden = self.dropout(hidden)
            if self.gru == 1:
                hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
                hidden = hidden.squeeze(0)

            if explain:
                alpha_all.append(alpha.detach().cpu().numpy())
                edges_all.append(edges.detach().cpu().numpy())
            
        if self.bert == 1:
            h_sub = entity_pretrain_emb[q_sub[nodes[:,0]]].cuda()
            hidden = hidden + h_sub

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda() # non_visited entities have scores of zeros
        scores_all[[nodes[:,0], nodes[:,1]]] = scores
        return scores_all, alpha_all, edges_all

    def loss(self, scores, labels): 
        scores_gene = scores[:, self.gene_idx]
        pos_scores = scores[[np.arange(len(labels)), labels]]
        max_n = torch.max(scores_gene, 1, keepdim=True)[0]
        base_loss = torch.sum(-pos_scores + max_n.squeeze() + torch.log(torch.sum(torch.exp(scores_gene - max_n),1)))
        return base_loss


