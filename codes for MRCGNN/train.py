import copy
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve,auc
import os
import random
import copy


import numpy as np
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)
#def train_model(model, optimizer, data_o, data_s, data_a, train_loader, val_loader, test_loader, args):
def train_model(model, optimizer, data_o,data_s,brics_feat, train_loader, val_loader, test_loader, args):
    m = torch.nn.Sigmoid()
    loss_fct=torch.nn.CrossEntropyLoss()
    b_xent = nn.BCEWithLogitsLoss()
    loss_history = []
    max_auc = 0
    max_f1=0
    num_class = model.mlp[-1].out_features

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')
        data_s.to('cuda')
        #data_a.to('cuda')

    # Train model
    lbl_1 = torch.ones(data_o.x.shape[0], 1)
    lbl_2 = torch.zeros(data_o.x.shape[0], 1)
    lbl = torch.cat((lbl_1, lbl_2), 1)
    if args.cuda:
        lbl = lbl.cuda()

    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    stoping = 0

    train_losses_epoch = []
    val_losses_epoch = []

    for epoch in range(args.epochs):
        #stoping=0
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        batch_losses = []

        for i, (inp) in enumerate(train_loader):

            label=inp[2]
            label=np.array(label,dtype=np.int64)
            label=torch.from_numpy(label)
            if args.cuda:
                label = label.cuda()

            model.train()
            optimizer.zero_grad()
            output, cla_os, cla_os_a, _ = model(data_o, data_s, brics_feat, inp)
            # output, disc_logits, nf_link_loss= model(data_o, data_s,brics_feat, inp)

            log = torch.squeeze(output)

            loss1 = loss_fct(log, label.long())#主任务LOSS
            # loss2 = b_xent(disc_logits, lbl)#对比学习LOSS
            # loss3 = torch.abs(sum(nf_link_loss))#正则化


            loss_train = loss1(log, label.long())
            # loss_train = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3

            loss_history.append(loss_train.cpu().detach().numpy())

            batch_losses.append(loss_train.item())

            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))
        
        train_losses_epoch.append(np.mean(batch_losses))



        y_pred_train1 = []
        y_label_train = np.array(y_label_train)
        y_pred_train = np.array(y_pred_train).reshape((-1, num_class))
        for i in range(y_pred_train.shape[0]):
            a = np.max(y_pred_train[i])
            for j in range(y_pred_train.shape[1]):
                if y_pred_train[i][j] == a:
                    #print(y_pred_train[i][j])
                    y_pred_train1.append(j)
                    break

        acc = accuracy_score(y_label_train, y_pred_train1)
        f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
        recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
        precision1 = precision_score(y_label_train, y_pred_train1, average='macro')


#
        if not args.fastmode:
            #acc_val, f1_val, recall_val,precision_val, loss_val = test(model, val_loader, data_o, data_s, data_a, args,0)
            acc_val, f1_val, recall_val,precision_val, loss_val = test(model, val_loader, data_o,data_s,brics_feat,  args,0)
            
            # if acc_val >= max_auc and f1_val>=max_f1:
            #     model_max = copy.deepcopy(model)
            #     max_auc = acc_val
            #     max_f1=f1_val
            #     stoping=0
            # else:
            #     stoping=stoping+1

            val_losses_epoch.append(loss_val.item())

            if f1_val > max_f1: 
                model_max = copy.deepcopy(model)
                max_f1 = f1_val
                max_auc = acc_val 
                stoping = 0
                print(f"--> 发现更好的 F1 Score: {max_f1:.4f}, 模型已更新")
            else:
                stoping += 1

            print('epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(acc),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val),
                  'f1_val: {:.4f}'.format(f1_val),
                  'recall_val: {:.4f}'.format(recall_val),
                  'precision_val: {:.4f}'.format(precision_val),
                  'time: {:.4f}s'.format(time.time() - t))
        else:#
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()



    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    plt.figure(figsize=(10, 6))
    
    # 绘制训练集损失
    epochs_range = range(1, len(train_losses_epoch) + 1)
    plt.plot(epochs_range, train_losses_epoch, label='Train Loss', color='blue', linewidth=2)
    
    # 绘制验证集损失
    if not args.fastmode:
        plt.plot(epochs_range, val_losses_epoch, label='Validation Loss', color='red', linewidth=2, linestyle='--')

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Testing
    #acc_test, f1_test, recall_test,precision_test, loss_test= test(model_max, test_loader, data_o, data_s, data_a, args,1)
    acc_test, f1_test, recall_test,precision_test, loss_test= test(model_max, test_loader, data_o,data_s,brics_feat, args,1)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'acc_test: {:.4f}'.format(acc_test),
          'f1_test: {:.4f}'.format(f1_test), 'precision_test: {:.4f}'.format(precision_test),'recall_test: {:.4f}'.format(recall_test))


#def test(model, loader, data_o, data_s, data_a, args,printfou):
def test(model, loader, data_o,data_s,brics_feat, args,printfou):

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.CrossEntropyLoss()
    b_xent = nn.BCEWithLogitsLoss()
    model.eval()
    y_pred = []
    y_label = []
    num_class = model.mlp[-1].out_features

    lbl_1 = torch.ones(data_o.x.shape[0], 1)
    lbl_2 = torch.zeros(data_o.x.shape[0], 1)
    lbl = torch.cat((lbl_1, lbl_2), 1)
    if args.cuda:
        lbl = lbl.cuda()

    zhongzi=args.zhongzi
    with torch.no_grad():
        for i, (inp) in enumerate(loader):
            label = inp[2]
            label = np.array(label, dtype=np.int64)
            label = torch.from_numpy(label)
            if args.cuda:
                label = label.cuda()

            output, cla_os, cla_os_a, _ = model(data_o, data_s, brics_feat, inp)
            # output,disc_logits, nf_link_loss= model(data_o,data_s,brics_feat,inp)
            #log = torch.squeeze(m(output))

            log = torch.squeeze(output)

            loss1 = loss_fct(log, label.long())
            # loss2 = b_xent(disc_logits, lbl)#对比学习LOSS
            # loss3 = torch.abs(sum(nf_link_loss))#正则化
            # loss = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3
            loss = loss1

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()

    y_pred_train1=[]
    y_label_train = np.array(y_label)
    y_pred_train = np.array(y_pred).reshape((-1, num_class))
    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                y_pred_train1.append(j)
                break
    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
    precision1 = precision_score(y_label_train, y_pred_train1, average='macro')
    y_label_train1 = np.zeros((y_label_train.shape[0], num_class))
    for i in range(y_label_train.shape[0]):
        y_label_train1[i][y_label_train[i]] = 1

    auc_hong=0
    aupr_hong=0
    nn1 = y_label_train1.shape[1]
    for i in range(y_label_train1.shape[1]):

        if np.sum(y_label_train1[:, i].reshape((-1))) < 1:
            nn1 = nn1 - 1
            continue
        else:

            auc_hong = auc_hong + roc_auc_score(y_label_train1[:, i].reshape((-1)), y_pred_train[:, i].reshape((-1)))
            precision, recall, _thresholds = precision_recall_curve(y_label_train1[:, i].reshape((-1)),
                                                                    y_pred_train[:, i].reshape((-1)))
            aupr_hong = aupr_hong + auc(recall, precision)

    auc_macro = auc_hong / nn1
    aupr_macro = aupr_hong / nn1
    auc1 = roc_auc_score(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)), average='micro')
    precision, recall, _thresholds = precision_recall_curve(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)))
    aupr = auc(recall, precision)

    if printfou==1:
        with open(args.out_file, 'a') as f:


            f.write(str(zhongzi)+'  '+str(acc)+'   '+str(f1_score1)+'   '+str(recall1)+'   '+str(precision1)+'   '+str(auc1)+'   '+str(aupr)+'   '+str(auc_macro)+'   '+str(aupr_macro)+'\n')





    return acc,f1_score1,recall1,precision1,loss