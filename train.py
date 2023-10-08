import numpy as np
import time

from data import load_ASERTAIN
from utils import print_log

from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.models import HGNN, HGNNP
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from dhg.experiments import HypergraphVertexClassificationTask as Task

import torch.nn as nn


from dhgnn import DHGNN

from fc import FC


def run(device, X, lbl, train_mask, test_mask, val_mask, G, net, lr , weight_decay, n_epoch, model_name):

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(n_epoch):
        if epoch > best_epoch+200:
            break
        # train
        train(net, X, G, lbl, train_mask, optimizer, epoch, model_name, device)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res, _ = infer(net, X, G, lbl, val_mask, epoch, model_name, device)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res, all_outs = infer(net, X, G, lbl, test_mask, best_epoch, model_name, device, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    return res, all_outs


def train(net, X, A, lbls, train_idx, optimizer, epoch, model_name, device):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    
    if model_name == "FC":
        outs = net(X)   
        outs = outs[train_idx]
    elif model_name == "DHGNN":
        #ids: indices selected during train/valid/test, torch.LongTensor
        ids = [i for i in range(X.size()[0])]
        ids = torch.tensor(ids).long()[train_idx].to(device)
        outs = net(ids=ids, feats=X, edge_dict=A.e_list, G=A.H, ite=epoch, device=device)
    else:
        outs = net(X, A)
        outs = outs[train_idx]

    lbls = lbls[train_idx]
    loss = F.binary_cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, epoch, model_name, device, test=False):
    evaluator = Evaluator(["accuracy", "f1_score"])
    net.eval()
    if model_name == "FC":
        all_outs = net(X)
        outs = all_outs[idx]
    elif model_name == "DHGNN":
        ids = [i for i in range(X.size()[0])]
        ids = torch.tensor(ids).long().to(device)
        all_outs = net(ids=ids, feats=X, edge_dict=A.e_list, G=A.H, ite=epoch, device=device)
        outs = all_outs[idx]
    else:
        all_outs = net(X, A)
        outs = all_outs[idx]

    lbls = lbls[idx]
    lbls = torch.argmax(lbls, dim=1)
    outs = torch.argmax(outs, dim=1)
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res, all_outs


def select_model(feat_dimension, n_hidden_layers, n_classes, n_conv, model, drop_rate, he_dropout, adjacent_centers, clusters, k_structured, k_nearest, k_cluster, wu_kmeans, wu_struct):
        if model == "HGNN":
            return HGNN(feat_dimension, n_hidden_layers, n_classes, n_conv, use_bn=True, drop_rate=drop_rate, he_dropout=he_dropout)
        elif model == "HGNNP":
            return HGNNP(feat_dimension, n_hidden_layers, n_classes, use_bn=True, drop_rate=drop_rate, he_dropout=he_dropout)
        elif model == "FC":
            return FC(feat_dimension, n_classes)
        elif model == "DHGNN":
            n_layers = 2
            return DHGNN(dim_feat=feat_dimension,
            n_categories=n_classes,
            k_structured=k_structured,
            k_nearest=k_nearest,
            k_cluster=k_cluster,
            wu_knn=0,
            wu_kmeans=wu_kmeans,
            wu_struct=wu_struct,
            clusters=clusters,
            adjacent_centers=adjacent_centers,
            n_layers=n_layers,
            layer_spec=[feat_dimension for l in range(n_layers-1)],
            dropout_rate=drop_rate,
            has_bias=True,
            )
        

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    selected_modalities=[['ECG'], ['EEG'], ['EMO'], ['GSR'], ['ECG', 'EEG'], ['ECG', 'EMO'], ['ECG', 'GSR'], ['EEG', 'EMO'], ['EEG', 'GSR'], ['EMO', 'GSR'], ['ECG', 'EEG', 'EMO'], ['ECG', 'EEG', 'GSR'], ['ECG', 'EMO', 'GSR'], ['EEG', 'EMO', 'GSR'], ['ECG', 'EEG', 'EMO', 'GSR']]

    label = "valence"
    train_ratio = 70
    val_ratio = 15
    test_ratio = 15
    n_classes = 2
    n_epoch = 10000
    model_name = "DHGNN" #HGNN, HGNNP, NB, SVM
    fusion_model = "HGNNP"
    fuse_models = True
    use_attributes = True
    opti = False
    trials = 10

    k = 66 #4, 20   
    drop_rate = 0.37
    lr = 0.001 #0.01, 0.001
    weight_decay = 5e-4
    n_hidden_layers = 8 #8
    n_conv = 2
    he_dropout = 0.5


    final_acc = 0
    final_f1 = 0
    all_accs = [0 for m in selected_modalities]
    all_f1s = [0 for m in selected_modalities]

    print_log("model: " + model_name)

    for trial in range(trials):
        print_log("trial: " + str(trial))
        i = 0
        inputs = []
        accs = []
        for m in selected_modalities:

            adjacent_centers = 1
            clusters = 400
            drop_rate = 0.5
            k = 4
            k_cluster = 4 #64
            k_nearest = 4 #64
            k_structured = 8 #8
            wu_kmeans = 10
            wu_struct = 5
            weight_decay: 5 * 10 ** -4
            lr: 0.001




            print_log("loading data: " + str(m))
            X, Y, train_mask, test_mask, val_mask, sa, va, lpa, hpa = load_ASERTAIN(selected_modalities=m, label=label, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, trial=trial)
            model = select_model(feat_dimension=X.shape[1], n_hidden_layers=n_hidden_layers, n_classes=n_classes, model=model_name, n_conv=n_conv, drop_rate=drop_rate, he_dropout=he_dropout, adjacent_centers=adjacent_centers, clusters=clusters, k_cluster=k_cluster, k_nearest=k_nearest, k_structured=k_structured, wu_kmeans=wu_kmeans, wu_struct=wu_struct)

            X = torch.tensor(X).float()

            print_log("generating hypergraph: " + str(m))
            G = Hypergraph(X.size()[0], device=device)
            G.add_hyperedges_from_feature_kNN(X, k=k)

            # if use_attributes:
            #     for a in sa:
            #         G.add_hyperedges(a, group_name="subject_attributes_"+str(a))
            #     for a in va:
            #         G.add_hyperedges(a, group_name="video_attributes_"+str(a))
            #     z = 0
            #     for a in lpa:
            #         G.add_hyperedges(a, group_name="low_personality_attributes_"+str(z))
            #         z += 1

            #     z = 0
            #     for a in hpa:
            #         G.add_hyperedges(a, group_name="high_personality_attributes_"+str(z))
            #         z += 1

            Y = [[0,1] if e == 1 else [1,0] for e in Y]
            Y = torch.tensor(Y).float()
            train_mask = torch.tensor(train_mask)
            val_mask = torch.tensor(val_mask)
            test_mask = torch.tensor(test_mask)
            # X = torch.eye(G.num_v)

            G.to(device)
            X = X.to(device)
            Y = Y.to(device)


            # lr = lrs[i]
            # weight_decay = wds[i]
            res, out = run(device, X, Y, train_mask, test_mask, val_mask, G, model, lr , weight_decay, n_epoch, model_name)
            all_accs[i] += res['accuracy']
            all_f1s[i] += res['f1_score']
            accs.append(res['accuracy']**2)
            inputs.append(out)
            i += 1

        if fuse_models:
            print_log("fusing models with: " + fusion_model)

            k = 4 #4, 20   
            drop_rate = 0.5
            lr = 0.001 #0.01, 0.001
            weight_decay = 5*10**-4
            n_hidden_layers = 8 #8
            n_conv = 2
            he_dropout = 0.5

            if fusion_model!="FC":   
                G = Hypergraph(2088)
                i = 0

                # weight of attributes 
                accs.append(0.5**2)
                # normalize weights so their sum is 1
                weights = [float(i)/sum(accs) for i in accs]
                # weights = accs  
                average_weight_index = len(inputs)


                if use_attributes:
                    for a in sa:
                        G.add_hyperedges(a, group_name="attr", e_weight=weights[average_weight_index])
                    
                    for a in va:
                        G.add_hyperedges(a, group_name="attr", e_weight=weights[average_weight_index])

                    for a in lpa:
                        G.add_hyperedges(a, group_name="attr", e_weight=weights[average_weight_index])
                        i += 1

                    for a in hpa:
                        G.add_hyperedges(a, group_name="attr", e_weight=weights[average_weight_index])

                j = 0
                for i in inputs:
                    G.add_hyperedges_from_feature_kNN(i, k=k, group_name="modality_"+str(j), e_weight=weights[j])
                    j += 1

                inputs = torch.cat(inputs, 1)
                G.add_hyperedges_from_feature_kNN(inputs, k=k, group_name="modality_fusion", e_weight=weights[average_weight_index])
            
            else:
                inputs = torch.cat(inputs, 1)

            model = select_model(feat_dimension=inputs.size()[1], n_hidden_layers=n_hidden_layers, n_classes=n_classes, model=fusion_model, n_conv=n_conv, drop_rate=drop_rate, he_dropout=he_dropout, adjacent_centers=adjacent_centers, clusters=clusters, k_cluster=k_cluster, k_nearest=k_nearest, k_structured=k_structured, wu_kmeans=wu_kmeans, wu_struct=wu_struct)

            final_res, _ = run(device, inputs, Y, train_mask, test_mask, val_mask, G, model, lr , weight_decay, n_epoch, fusion_model)
            final_acc += final_res['accuracy']
            final_f1 += final_res['f1_score']


        print("acc: ", np.divide(all_accs,trials))
        print("f1: ", np.divide(all_f1s,trials))
        print(selected_modalities)

        if fuse_models:
            print("final acc: ", final_acc/trials)
            print("final f1: ", final_f1/trials)
            print("weights: ", weights)



    
