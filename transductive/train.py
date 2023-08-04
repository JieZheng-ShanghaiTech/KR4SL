import os
import argparse
import torch
import numpy as np
import time
import random
from load_data import DataLoader
from base_model import BaseModel

def get_params():
    parser = argparse.ArgumentParser(description="Parser for RED-GNN")
    parser.add_argument('--data_path', type=str, default='../data/transductive/')
    parser.add_argument('--early_stop', type=int, default=3, help='the early stop flag')
    parser.add_argument('--gpu', type=int, default=0, help='choose the gpu id')
    parser.add_argument('--seed', type=str, default=1234)
    parser.add_argument('--lr', type=float, default=0.0011)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=48)
    parser.add_argument('--attn_dim', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--explain', default=False)
    parser.add_argument('--suffix', type=str, default='trans')
    parser.add_argument('--time_num', type=int, default=0)
    # parser.add_argument('--constrain_go_depth', type=int, default=0, help='whether remove the GO term at shallow level of GO tree, 0: not remove, 1: remove by exclude_set1)  
    
    
    args = parser.parse_known_args()
    return args

class Options(object):
    pass

if __name__ == '__main__':
    args,_ = get_params()
    np.random.seed(args.seed)#+args.time_num
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options
    suffix = args.suffix
    print(suffix)
    if not os.path.exists('../results/'+suffix):
        os.makedirs('../results/'+suffix)
    opts.perf_file = os.path.join('../results/'+suffix,  suffix+'_perf_'+str(args.time_num)+'fold'+'.txt')
    opts.ckpt_file = os.path.join('../results/'+suffix,  suffix+'_ckpt_model_'+str(args.time_num)+'fold'+'.pkl')
    opts.best_1fold_file = os.path.join('../results/'+suffix, suffix+'_best_'+str(args.time_num)+'fold_model'+'.pkl')

    torch.cuda.set_device(args.gpu)
    print('gpu:', args.gpu)

    
    opts.lr = args.lr
    opts.decay_rate = 0.9938
    opts.lamb = 0.000089
    opts.hidden_dim = args.hidden_dim
    opts.attn_dim = args.attn_dim
    opts.n_layer = args.n_layer
    # opts.dropout = args.dropout
    opts.act = 'relu'
    opts.n_batch = args.batch_size
    opts.n_tbatch = 50
    opts.load_ckpt = False  
    opts.explain = args.explain

    opts.seed = args.seed+args.time_num
    loader = DataLoader(args.data_path, opts)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    config_str = '\n%.4f, %.4f, %.6f, %d, %d, %d, %d, %s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)
    import ipdb

    all_result = {}
    time_num = args.time_num
    print('-----------Training for time %d-----------' %(time_num))
    with open(opts.perf_file, 'a+') as f:
        f.write('-----------Training for time %d-----------\n' %(time_num))

    model = BaseModel(opts, loader)
    best_n50 = 0
    early_stop_cnt = 0
    start = 0
    if opts.load_ckpt:
            # start = 12
            print('Loading best model and checkpoint ...')
            model.model.load_state_dict(torch.load(opts.best_file))
            best_p50_fold, out_str_val, _ = model.evaluate(data='valid')
            _, out_str_tr, _ = model.evaluate(data='train')
            _, out_str_tst, _ = model.evaluate(data='test')
            model.model.load_state_dict(torch.load(opts.best_1fold_file))
            train_mrr, out_str_tr, result_tr = model.evaluate(data='train')
            best_n50, out_str_val, result_val = model.evaluate(data='valid')
            tst_n50, out_str_tst, best_result = model.evaluate(data='test')
            best_str = out_str_tr + out_str_val + out_str_tst
            best_model = model.model
            print(best_str)
            model.model.load_state_dict(torch.load(opts.ckpt_file))
    
    for epoch in range(start, 15):
        model.train_batch()
        torch.save(model.model.state_dict(), opts.ckpt_file)
        if (epoch + 1) % 3 == 0:
            i_time = time.time()
            train_mrr, out_str_tr, result_tr = model.evaluate(data='train')
            val_n50, out_str_val, result_val = model.evaluate(data='valid')
            tst_n50, out_str_tst, result_tst = model.evaluate(data='test')
            i_time = time.time() - i_time
            time_str = '[TIME] train:%.4f inference:%.4f\n' %(model.t_time, i_time)
            out_str = str(epoch) + '\t' + out_str_tr + out_str_val + out_str_tst + time_str
            print(out_str)
            with open(opts.perf_file, 'a+') as f:
                f.write(out_str)
            if val_n50 > best_n50:
                best_n50 = val_n50
                best_str = out_str
                best_result = result_tst
                best_model = model.model
                torch.save(best_model.state_dict(), opts.best_1fold_file)
                with open(opts.perf_file, 'a+') as f:
                    f.write('epoch {} is the best!\n\n'.format(epoch))
            else:
                early_stop_cnt += 1
                if early_stop_cnt == args.early_stop:
                    print('Early stopping triggered!')
                    break
    print(best_str)
    print('\n')
    with open(opts.perf_file, 'a+') as f:
        f.write('the best result:\n')
        f.write(best_str)
    all_result[time_num] = best_result
    np.save('results/'+suffix+'/all_result_'+str(time_num)+'fold.npy', all_result)

    

