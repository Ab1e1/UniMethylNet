import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, precision_score, recall_score
import shutil
import os
import torch.nn.functional as F
import time

import sys
from data_load import *
from model_torch import *
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_pred_prob):
    # 将预测概率转换为离散标签
    y_pred = (np.array(y_pred_prob) >= 0.5).astype(int)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 灵敏度 (Sensitivity)
    SN = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    # 特异性 (Specificity)
    SP = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

    # 准确率 (Accuracy)
    ACC = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0

    # 马修斯相关系数 (MCC)
    MCC = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-10) * 100

    # AUC 计算
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    AUC = auc(fpr, tpr) * 100

    # F1 分数
    F1 = f1_score(y_true, y_pred) * 100

    # 召回率 (Recall)
    Recall = recall_score(y_true, y_pred) * 100

    # 精度 (Precision)
    Pre = precision_score(y_true, y_pred) * 100

    return SN, SP, ACC, MCC, AUC, F1, Recall, Pre

def train(model, loss, optimizer, x_val, y_val, args):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    model.train()

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x, args)

    if args.if_bce == 'Y':
        pred_prob = torch.sigmoid(fx)
        output = loss(pred_prob.squeeze(), y.type(torch.FloatTensor))
    else:
        pred_prob = torch.log_softmax(fx, dim=1)
        output = loss(fx, y)

    # Backward
    output.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    # Update parameters
    optimizer.step()

    return output.item(), pred_prob, list(np.array(y_val))

def predict(model, x_val, args):
    model.eval()  # evaluation mode do not use drop out
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x, args)
    return output



def save_checkpoint(state,is_best,model_path):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(state, model_path + '/' + 'checkpoint.pth.tar')

    else:
        print("=> Validation Performance did not improve")


def ytest_ypred_to_file(y_test, y_pred, out_fn):
    with open(out_fn,'w') as f:
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred[i])+'\n')



if __name__ == '__main__':

    #torch.manual_seed(1000)

    parser = argparse.ArgumentParser()

    # main option
    parser.add_argument("-m", "--mode", action="store", dest='mode', required=True,choices=['cnn','cnn-rnn'],
                        help="mode")

    parser.add_argument("-pos_fa", "--positive_fasta", action="store", dest='pos_fa', required=True,
                        help="positive fasta file")
    parser.add_argument("-neg_fa", "--negative_fasta", action="store", dest='neg_fa', required=True,
                        help="negative fasta file")

    parser.add_argument("-od", "--out_dir", action="store", dest='out_dir', required=True,
                        help="output directory")

    # cnn option
    parser.add_argument("-fltnum", "--filter_num", action="store", dest='filter_num', default='256-256-256-256-256',
                        help="filter number")
    parser.add_argument("-fltsize", "--filter_size", action="store", dest='filter_size', default='10-10-10-10-10',
                        help="filter size")
    parser.add_argument("-pool", "--pool_size", action="store", dest='pool_size', default=0, type=int,
                        help="pool size")
    parser.add_argument("-cnndrop", "--cnndrop_out", action="store", dest='cnndrop_out', default=0.5, type=float,
                        help="cnn drop out")
    parser.add_argument("-bn", "--if_bn", action="store", dest='if_bn', default='Y',
                        help="if batch normalization")

    # rnn option
    parser.add_argument("-rnnsize", "--rnn_size", action="store", dest='rnn_size', default=32, type=int,
                        help="rnn size")


    # fc option
    parser.add_argument("-fc", "--fc_size", action="store", dest='fc_size', default=0.0, type=float,
                        help="fully connected size")



    # optimization option
    parser.add_argument("-bce", "--if_bce", action="store", dest='if_bce', default='Y',
                        help="if use BCEloss function")
    parser.add_argument("-maxepc", "--max_epochs", action="store", dest='max_epochs', default=30, type=int,
                        help="max epochs")
    parser.add_argument("-lr", "--learning_rate", action="store", dest='learning_rate', default=0.01, type=float,
                        help="learning rate")
    parser.add_argument("-lrstep", "--lr_decay_step", action="store", dest='lr_decay_step', default=10, type=int,
                        help="learning rate decay step")
    parser.add_argument("-lrgamma", "--lr_decay_gamma", action="store", dest='lr_decay_gamma', default=0.5, type=float,
                        help="learning rate decay gamma")
    parser.add_argument("-gdclp", "--grad_clip", action="store", dest='grad_clip', default=5, type=int,
                        help="gradient clip value magnitude")
    parser.add_argument("-patlim", "--patience_limit", action="store", dest='patience_limit', default=5, type=int,
                        help="patience")
    parser.add_argument("-batch", "--batch_size", action="store", dest='batch_size', default=256, type=int,
                        help="batch size")




    args = parser.parse_args()


    #load m6A data

    #data_path = args.data_dir
    pos_train_fa = args.pos_fa
    neg_train_fa = args.neg_fa

    wordvec_len=4

    X_train, y_train, X_test, y_test = load_train_val_bicoding(pos_train_fa,neg_train_fa)
    X_train, y_train, X_test, y_test = load_in_torch_fmt(X_train, y_train, X_test, y_test, wordvec_len)




    model_dir_base = args.filter_num + '_' + args.filter_size + '_' + str(
                args.pool_size) + '_' + str(args.cnndrop_out) \
                        + '_' + args.if_bn + '_' +args.if_bce + '_' +\
        str(args.fc_size) + '_' + str(args.learning_rate) + '_' + str(args.batch_size)

    n_classes = 2
    n_examples = len(X_train)
    loss = torch.nn.CrossEntropyLoss(size_average=False)

    if args.if_bce == 'Y':
        n_classes = n_classes - 1
        loss = torch.nn.BCELoss(size_average=False)

    if args.mode == 'cnn':
        model = ConvNet(output_dim=n_classes, args=args,wordvec_len=wordvec_len)
        model_dir = args.mode + '/' + model_dir_base

    if args.mode == 'cnn-rnn':
        filter_num = 6  # 或者根据卷积层的实际输出通道数设置
        feature = 256     # 根据卷积层的输入通道数设置
        seq_len = 256   # 根据实际的序列长度设置

        model = ConvNet_BiLSTM(output_dim=n_classes, args=args, wordvec_len=wordvec_len, filter_num=filter_num, feature=feature, seq_len=seq_len)
        
        #打印模型结构
        #print(model)

        model_dir = args.mode + '/' +model_dir_base +'_'+str(args.rnn_size)


    model_path = args.out_dir + '/' + model_dir
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    # 创建 result 文件夹
    # 获取out_dir的上一级目录路径
    result_dir = os.path.join(os.path.dirname(args.out_dir), 'result_paper')

# 如果result文件夹不存在，则创建
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    print("> model_dir:",model_path)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    batch_size = args.batch_size
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)


    best_acc=0
    patience=0
    train_losses = []
    test_losses = []
    
    best_acc_train = 0
    best_epoch_train = 0
    best_acc_test = 0
    best_epoch_test = 0

    for i in range(args.max_epochs):

        start_time = time.time()
        scheduler.step()
        
         # Initialize training loss for the current epoch
        train_loss = 0.

        cost = 0.
        y_pred_prob_train = []
        y_batch_train = []

        num_batches = n_examples // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            output_train,y_pred_prob,y_batch=train(model, loss, optimizer, X_train[start:end], y_train[start:end],args)
            cost += output_train
            train_loss += output_train  # accumulate the loss

            prob_data = y_pred_prob.data.numpy()
            if args.if_bce == 'Y':
                for m in range(len(prob_data)):
                    y_pred_prob_train.append(prob_data[m][0])

            else:
                for m in range(len(prob_data)):
                    y_pred_prob_train.append(np.exp(prob_data)[m][1])

            y_batch_train+=y_batch

        #rest samples
        start, end=num_batches * batch_size, n_examples
        output_train, y_pred_prob, y_batch = train(model, loss, optimizer, X_train[start:end], y_train[start:end], args)
        cost += output_train
        train_loss += output_train

        prob_data = y_pred_prob.data.numpy()
        if args.if_bce == 'Y':
            for m in range(len(prob_data)):
                y_pred_prob_train.append(prob_data[m][0])

        else:
            for m in range(len(prob_data)):
                y_pred_prob_train.append(np.exp(prob_data)[m][1])


        y_batch_train += y_batch

        # 计算训练指标
        SN, SP, ACC, MCC, AUC_train, F1, Recall, Pre = compute_metrics(np.array(y_batch_train), np.array(y_pred_prob_train))
        #train AUC
        fpr_train, tpr_train, thresholds_train = roc_curve(y_batch_train, y_pred_prob_train)
        
        # 计算训练集的准确率
        cur_acc_train = 100. * np.mean((np.array(y_pred_prob_train) >= 0.5).astype(int) == np.array(y_batch_train))

        # 更新最佳训练准确率及对应的 epoch
        if cur_acc_train > best_acc_train:
            best_acc_train = cur_acc_train
            best_epoch_train = i + 1
        # Append the average loss for this epoch
        train_losses.append(train_loss / num_batches)



        #predict test
        output_test = predict(model, X_test,args)
        test_loss = 0.
        y_pred_prob_test = []

        if args.if_bce == 'Y':
            y_pred_test=[]
            prob_data=F.sigmoid(output_test).data.numpy()
            for m in range(len(prob_data)):
                y_pred_prob_test.append(prob_data[m][0])
                if prob_data[m][0]>=0.5:
                    y_pred_test.append(1)
                else:
                    y_pred_test.append(0)
            test_loss += loss(F.sigmoid(output_test).squeeze(), torch.FloatTensor(y_test)).item()
        else:
            y_pred_test=output_test.data.numpy().argmax(axis=1)
            prob_data =F.log_softmax(output_test).data.numpy()
            for m in range(len(prob_data)):
                y_pred_prob_test.append(np.exp(prob_data)[m][1])
            test_loss += loss(output_test, torch.LongTensor(y_test)).item()
        
        # Append the average test loss for this epoch
        test_losses.append(test_loss)


        #test AUROC
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
        SN_test, SP_test, ACC_test, MCC_test, AUC_test, F1_test, Recall_test, Pre_test = compute_metrics(y_test, y_pred_prob_test)


        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)



        print("Epoch %d, cost = %f, AUROC_train = %0.3f, acc = %.2f%%, AUROC_test = %0.3f"
              % (i + 1, cost / num_batches, auc(fpr_train, tpr_train),100. * np.mean(y_pred_test == y_test),auc(fpr_test, tpr_test)))
        print("time cost: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))



        
        cur_acc_test=100. * np.mean(y_pred_test == y_test)
        # 更新最佳测试准确率及对应的 epoch
        if cur_acc_test > best_acc_test:
            best_acc_test = cur_acc_test
            best_epoch_test = i + 1
        is_best = bool(cur_acc_test >= best_acc_test)
        best_acc = max(cur_acc_test, best_acc_test)
        save_checkpoint({
            'epoch': i+1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_acc_test,
            'optimizer': optimizer.state_dict()
        }, is_best,model_path)

        # 保存指标到文件
        metrics_file = os.path.join(result_dir, 'metrics_6mA_sapiens.txt')
        with open(metrics_file, 'a') as f:
            if i == 0:  # 文件开头写入模型参数量
                param_count = sum(p.numel() for p in model.parameters())
                f.write(f"Model Parameters: {param_count}\n")

            # 记录训练集指标
            f.write(f"Epoch {i + 1} (Train): SN={SN:.2f}, SP={SP:.2f}, ACC={ACC:.2f}, MCC={MCC:.2f}, AUC={AUC_train:.2f}, F1={F1:.2f}, Recall={Recall:.2f}, Pre={Pre:.2f}\n")
    
            # 记录测试集指标
            f.write(f"Epoch {i + 1} (Test): SN={SN_test:.2f}, SP={SP_test:.2f}, ACC={ACC_test:.2f}, MCC={MCC_test:.2f}, AUC={AUC_test:.2f}, F1={F1_test:.2f}, Recall={Recall_test:.2f}, Pre={Pre_test:.2f}\n")


    # 在训练结束后，将最佳训练和测试准确率及其对应的 epoch 写入文件末端
    with open(metrics_file, 'a') as f:
        f.write(f"\nBest Training Accuracy: {best_acc_train:.2f}% at Epoch {best_epoch_train}\n")
        f.write(f"Best Test Accuracy: {best_acc_test:.2f}% at Epoch {best_epoch_test}\n")
        # #patience
        # if not is_best:
        #     patience+=1
        #     if patience>=args.patience_limit:
        #         break

        # else:
        #     patience=0




        # if is_best:
        #     ytest_ypred_to_file(y_batch_train, y_pred_prob_train,
        #                         model_path + '/' + 'predout_train.tsv')

        #     ytest_ypred_to_file(y_test, y_pred_prob_test,
        #                         model_path + '/' + 'predout_val.tsv')




    print('> best acc:',best_acc_test)

# 保存 loss 曲线到 PNG 文件
def save_loss_plot(train_losses, test_losses, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图像为 PNG 文件
    output_path = os.path.join(output_dir, 'loss_curve_thaliana.png')
    plt.savefig(output_path)
    plt.close()  # 关闭当前图表，释放内存
    print(f"Loss curve saved at {output_path}")

# 假设保存路径为 result 文件夹
result_dir = os.path.join(os.path.dirname(args.out_dir), 'result_paper')
save_loss_plot(train_losses, test_losses, result_dir)






