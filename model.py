import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss_and_optimizer(model, args):
    """Return loss and optimizer"""
    if args.loss_name == 'BCELoss':
        loss = torch.nn.BCELoss()
    elif args.loss_name == 'weightedloss':
        loss = weightedloss()
    elif args.loss_name == 'BCEFocalLoss':
        loss = BCEFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)#, amsgrad=True)
    return loss, optimizer

class weightedloss(nn.Module):
    def __init__(self, cls_num_list=[], smooth_head=0.1, smooth_tail=0.2, shape='concave', power=None):
        super(weightedloss, self).__init__()
        self.smooth =  [0.3, 0.7] # 0, 1分类

    def forward(self, x, target):
        weight = torch.zeros_like(target).float().cuda()
        weight = torch.fill_(weight,self.smooth[0])
        weight[target>0]=self.smooth[1]
        loss = nn.BCELoss(weight=weight)(x,target.float())
        return loss.mean()

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.smooth =  [1, 2] # 0, 1分类

    def forward(self, predict, target):
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        weight = torch.zeros_like(target).float().cuda()
        weight = torch.fill_(weight,self.smooth[0])
        weight[target>0]=self.smooth[1]
        # loss=-self.alpha*(1-predict)**self.gamma*target*torch.log(predict)-(1-self.alpha)*predict**self.gamma*(1-target)*torch.log(1-predict)
        # loss=-weight*(1-predict)**self.gamma*target*torch.log(predict)-(1-weight)*predict**self.gamma*(1-target)*torch.log(1-predict)
        loss=-weight*predict**self.gamma*target*torch.log(predict)-weight*(1-predict)**self.gamma*(1-target)*torch.log(1-predict)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def get_model(LOGGER, args):
    if args.model_name == 'LSTM':
        model = CricketLSTMModel(length_seq=args.length_seq)
    elif args.model_name == 'Linear':
        model = CricketLinearModel(length_seq=args.length_seq)
    elif args.model_name == '1D':
        model = Cricket1DModel()
    elif args.model_name == 'PEM':
        model
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    if args.resume:
        if LOGGER:
            LOGGER.info("Loading model parameters from {}".format(args.resume))
        else:
            print("Loading model parameters from {}".format(args.resume))
        model.load_state_dict(
            torch.load(args.resume, map_location=torch.device(device))
        )
    return model.to(device)

class CricketLSTMModel(torch.nn.Module):
    """
    LSTM模型
    """
    def __init__(self, length_seq=50):
        super().__init__()
        self.d_input=2048
        self.n_input = 75
        self.dropout=0.2
        # 降维2048->256
        self.layer1 = torch.nn.Sequential(
            nn.Linear(self.d_input, self.d_input//4),
            # nn.BatchNorm1d(self.n_input),
            torch.nn.LayerNorm([self.d_input//4]),
            nn.GELU(),
        )
        # 时序特征提取
        self.LSTM = nn.LSTM(self.d_input//4, hidden_size=self.d_input//8, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=self.dropout)
        # 分类
        self.classifier = torch.nn.Sequential(
            nn.Linear(self.d_input//8, self.d_input//32),
            nn.LeakyReLU(True),
            nn.Linear(self.d_input//32, 1)
        )

    def forward(self, x):
        # 输入b, L, 2048
        # L,在训练时为75
        x = self.layer1(x)
        x,_ = self.LSTM(x)
        x = x[:, :, :self.d_input//8] + x[:, :, self.d_input//8:]
        x = self.classifier(x)
        x = F.sigmoid(x)
        # print(x.shape)
        return x.squeeze(dim=2)

class CricketLinearModel(torch.nn.Module):
    """
    单张照片进行分类操作
    Linear模型
    """

    def __init__(self, length_seq=1):
        super().__init__()
        self.d_input=2048
        self.layer1 = torch.nn.Sequential(
            nn.Linear(self.d_input, self.d_input//2),
            torch.nn.BatchNorm1d(self.d_input//2),
            nn.GELU(),
        )
        self.layer2 = torch.nn.Sequential(
            nn.Linear(self.d_input//2, self.d_input//4),
            torch.nn.BatchNorm1d(self.d_input//4),
            nn.GELU(),
        )
        self.layer3 = torch.nn.Sequential(
            nn.Linear(self.d_input//4, self.d_input//8),
            torch.nn.BatchNorm1d(self.d_input//8),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        self.classifier = torch.nn.Linear(self.d_input//8, 1)

    def forward(self, x):
        # 输入b, L, 2048
        # L,在训练时为75
        x = x.squeeze()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        x = F.sigmoid(x)

        return x

class Cricket1DModel(torch.nn.Module):
    """
    1D模型
    """

    def __init__(self, length_seq=1):
        super().__init__()
        self.d_input=2048
        self.layer1 = torch.nn.Sequential(
            nn.Linear(self.d_input, self.d_input//8),
            torch.nn.LayerNorm([self.d_input//8]),
            nn.GELU(),
        )
        self.layer2 = torch.nn.Sequential(
            nn.Conv1d(in_channels=self.d_input//8, out_channels=self.d_input//4, kernel_size=11, padding=5),
            # torch.nn.BatchNorm1d(self.d_input//4),
            # torch.nn.LayerNorm([self.d_input//4]),
            nn.GELU(),
        )
        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(in_channels=self.d_input//4, out_channels=self.d_input//8, kernel_size=11, padding=5),
            # torch.nn.BatchNorm1d(self.d_input//8),
            # torch.nn.LayerNorm([self.d_input//8]),
            nn.GELU(),
            nn.Conv1d(in_channels=self.d_input//8, out_channels=self.d_input//16, kernel_size=11, padding=5),
            # torch.nn.BatchNorm1d(self.d_input//8),
            # torch.nn.LayerNorm([self.d_input//8]),
            nn.GELU(),
        )
        self.classifier = torch.nn.Sequential(
            nn.Linear(self.d_input//16, self.d_input//32),
            nn.LeakyReLU(True),
            nn.Linear(self.d_input//32, 1)
        )

    def forward(self, x):
        # 输入b, L, 2048
        # L,在训练时为75
        x = self.layer1(x)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        x = self.classifier(x)
        x = F.sigmoid(x).squeeze(dim=2)

        return x