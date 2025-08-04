#torch version: 0.4.0a0+0e24630

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from cbam import CBAM
from cpca import RepBlock




# We need to create two sequential models here since PyTorch doesn't have nn.View()
class ConvNet(torch.nn.Module):
    def __init__(self, output_dim,args,wordvec_len):
        super(ConvNet, self).__init__()

        self.conv = torch.nn.Sequential()

        #cnn layers
        for i in range(len(args.filter_size.split('-'))):

            if i==0:
                self.conv.add_module("conv_" + str(i+1),
                                     torch.nn.Conv1d(wordvec_len, int(args.filter_num.split('-')[i]),
                                                     kernel_size=int(args.filter_size.split('-')[i]),
                                                     padding=int(int(args.filter_size.split('-')[i])/2)))


            else:
                self.conv.add_module("conv_" + str(i+1),
                                     torch.nn.Conv1d(int(args.filter_num.split('-')[i-1]), int(args.filter_num.split('-')[i]),
                                                     kernel_size=int(args.filter_size.split('-')[i])))

            # pool
            if args.pool_size != 0:
                self.conv.add_module("maxpool_" + str(i + 1), torch.nn.MaxPool1d(kernel_size=args.pool_size))


            #activation
            self.conv.add_module("relu_"+str(i+1), torch.nn.ReLU())

            # batch norm
            if args.if_bn == 'Y':
                self.conv.add_module("batchnorm_" + str(i + 1),
                                     torch.nn.BatchNorm1d(int(args.filter_num.split('-')[i])))


            #dropout
            self.conv.add_module("dropout_"+str(i+1), torch.nn.Dropout(args.cnndrop_out))




        #fc layer
        self.fc = torch.nn.Sequential()
        if args.fc_size >0:
            self.fc.add_module("fc_1", torch.nn.Linear(int(args.filter_num.split('-')[-1]), int(args.fc_size)))
            self.fc.add_module("relu_1", torch.nn.ReLU())
            self.fc.add_module("fc_2", torch.nn.Linear(int(args.fc_size), output_dim))
        else:
            self.fc.add_module("fc_1", torch.nn.Linear(int(args.filter_num.split('-')[-1]), output_dim))



    def forward(self, x, args):
        x = x.transpose(1, 2)   #convert to 4*101
        x = self.conv.forward(x)
        x=torch.max(x,2)[0]
        x = x.view(-1,int(args.filter_num.split('-')[-1]))

        return self.fc.forward(x)





class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, _ = x.shape
        # Split embedding into multiple heads
        values = self.values(x).view(N, seq_length, self.heads, self.head_dim)
        keys = self.keys(x).view(N, seq_length, self.heads, self.head_dim)
        queries = self.queries(x).view(N, seq_length, self.heads, self.head_dim)

        # Transpose to get dimensions N, heads, seq_length, head_dim
        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Calculate energy scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, seq_length, seq_length)
        attention = F.softmax(energy, dim=3)  # (N, heads, seq_length, seq_length)

        # Weighted sum of values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_length, self.embed_size)
        
        return self.fc_out(out)


class Position_linear(nn.Module):
    def __init__(self, window_size=3, filter_num=6, feature=256, seq_len=256):
        super(Position_linear, self).__init__()
        self.filter_num = filter_num
        self.feature = feature
        self.window_size = window_size
        self.seq_len = seq_len
        self.pad_len = int(self.window_size / 2)

        # Create a list to store linear layers for each position
        self.dense_layer_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.window_size * self.feature, self.filter_num),  # Ensure this matches the reshaped window
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            for _ in range(self.seq_len)
        ])

    def forward(self, inputs):
    # Transpose input for convolution operation
        x = torch.transpose(inputs, 1, 2)
    # Pad the sequence with zeros to handle border effects in convolution
        x = F.pad(x, (self.pad_len, self.pad_len), "constant", 0)
    # Transpose back to the original shape
        x = torch.transpose(x, 1, 2)

    # Temporary list to store intermediate results for each position
        x_temp = []
        for i in range(self.pad_len, self.seq_len + self.pad_len):
        # Extract window
            window = x[:, i - self.pad_len: i + self.pad_len + 1, :]
        
        # Check if the window has the correct size
            if window.size(1) != self.window_size:
            # Print a warning and adjust window size (can be zero padding)
                #print(f"Adjusting window size: {window.size(1)} instead of {self.window_size}")
                window = F.pad(window, (0, 0, 0, self.window_size - window.size(1)))
        
        # Print the shape for debugging
            #print(f"Window shape: {window.shape}")
        
        # Reshape and apply linear layer
            reshaped_window = window.reshape(window.size(0), -1)
            #print(f"Reshaped window shape: {reshaped_window.shape}")

        # Add assert to check if the reshaped window has the correct size
            expected_size = self.window_size * self.feature
            assert reshaped_window.size(1) == expected_size, \
                f"Expected {expected_size}, but got {reshaped_window.size(1)}"

        # Pass through linear layer
            x_temp.append(
                self.dense_layer_net[i - self.pad_len](reshaped_window)
            )

    # Stack the intermediate results to get the final output
        x = torch.stack(x_temp)
        x = torch.transpose(x, 0, 1)
        return x





class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x



class ConvNet_BiLSTM(torch.nn.Module):
    def __init__(self, output_dim, args, wordvec_len, filter_num, feature, seq_len):
        super(ConvNet_BiLSTM, self).__init__()

        self.conv = torch.nn.Sequential()

        # cnn layers
        for i in range(len(args.filter_size.split('-'))):
            if i == 0:
                self.conv.add_module("conv_" + str(i + 1),
                                     torch.nn.Conv1d(wordvec_len, int(args.filter_num.split('-')[i]),
                                                     kernel_size=int(args.filter_size.split('-')[i]),
                                                     padding=int(int(args.filter_size.split('-')[i]) / 2)
                                                     ))
            else:
                self.conv.add_module("conv_" + str(i + 1),
                                     torch.nn.Conv1d(int(args.filter_num.split('-')[i - 1]),
                                                     int(args.filter_num.split('-')[i]),
                                                     kernel_size=int(args.filter_size.split('-')[i])
                                                     ))

            # pool
            if args.pool_size != 0:
                self.conv.add_module("maxpool_" + str(i + 1), torch.nn.MaxPool1d(kernel_size=args.pool_size))

            # activation
            self.conv.add_module("relu_" + str(i + 1), torch.nn.ReLU())

            # batchnorm
            if args.if_bn == 'Y':
                self.conv.add_module("batchnorm_" + str(i + 1),
                                     torch.nn.BatchNorm1d(int(args.filter_num.split('-')[i])))

            # dropout
            self.conv.add_module("dropout_" + str(i + 1), torch.nn.Dropout(args.cnndrop_out))

        # Ensure the filter_num matches the last conv layer's output channels
        self.position_linear_3 = Position_linear(window_size=3, filter_num=6, feature=256, seq_len=256)
        self.position_linear_5 = Position_linear(window_size=5, filter_num=6, feature=256, seq_len=256)
        self.position_linear_7 = Position_linear(window_size=7, filter_num=6, feature=256, seq_len=256)

        self.lstm = torch.nn.LSTM(int(args.filter_num.split('-')[-1]), args.rnn_size, 1, batch_first=True, bidirectional=True)
        
        # Print rnn_size to check
        #print("RNN Size:", args.rnn_size)
        # 初始化 CBAM
        #self.cbam = CBAM(in_channels=args.rnn_size * 2)# LSTM 是双向的，输出通道数是 rnn_size * 2
        self.rep_block = RepBlock(in_channels=args.rnn_size * 2, out_channels=args.rnn_size * 2)#RNN Size: 32
    
    #    # 自注意力层
    #     self.attention = SelfAttention(embed_size=args.rnn_size * 2, heads=8)  # 8 是示例，可以根据需要调整

        # fc layer
        self.fc = torch.nn.Sequential()
        if args.fc_size > 0:
            self.fc.add_module("fc_1", torch.nn.Linear(args.rnn_size * 2, int(args.fc_size)))
            self.fc.add_module("relu_1", torch.nn.ReLU())
            self.fc.add_module("fc_2", torch.nn.Linear(int(args.fc_size), output_dim))
        else:
            self.fc.add_module("fc_1", torch.nn.Linear(args.rnn_size * 2, output_dim))

    def forward(self, x, args):

        h0 = Variable(torch.zeros(1 * 2, x.size(0), args.rnn_size))  # 2 for bidirection
        c0 = Variable(torch.zeros(1 * 2, x.size(0), args.rnn_size))

        # Transpose input before convolution
        x = x.transpose(1, 2)
        #print(f"Input shape after transpose (before conv): {x.shape}")
        
        # Apply convolution
        x = self.conv(x)
        #print(f"Shape after conv layers: {x.shape}")
        
        # Transpose back for the next layers
        x = x.transpose(1, 2)
        #print(f"Shape after transpose (before Position_linear_3): {x.shape}")
        
        # Apply Position_linear layers
        x3 = self.position_linear_3(x)
        #print(f"Shape after position_linear_3: {x3.shape}")
        # Transpose back for the next layers
        x3 = x3.transpose(1, 2)
        #print(f"Shape after transpose (before Position_linear_5): {x3.shape}")
        x5 = self.position_linear_5(x3)
        #print(f"Shape after position_linear_5: {x5.shape}")
        x5 = x5.transpose(1, 2)
        x7 = self.position_linear_7(x5)
        #print(f"Shape after position_linear_7: {x7.shape}")
        x7 = x7.transpose(1, 2)
        # Forward propagate through LSTM
        out, _ = self.lstm(x7, (h0, c0))
        #print(f"Shape after LSTM: {out.shape}")
        #
        B, N, C = out.size()  # B=256, N=6, C=64
        H, W = 6, 1  # 选择合适的 H 和 W，这里假设 H=6, W=1
        out_reshape = out.reshape(B, C, H, W)
        # print(out_reshape.shape)
        # Apply CSDA
        out = self.rep_block(out_reshape)
        B,C,H,W=out.size()
        out=out.view(B,N,C)
        # print(x.shape)
        # Final fully connected layer
        out = self.fc(torch.mean(out, 1))
        #print(f"Shape after FC: {out.shape}")

        return out





