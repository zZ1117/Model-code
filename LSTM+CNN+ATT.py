import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 定义注意力层
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = torch.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs

# 定义CNN+LSTM模型
class ATCNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(ATCNNLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将输入数据的维度调整为(batch_size, input_size, sequence_length)
        out = self.conv1(x)
        out = self.relu(out)
        out = out.permute(0, 2, 1)  # 将维度调整回(batch_size, sequence_length, input_size)
        out, _ = self.lstm(out)
        out = self.attention(out)  # 注意力机制
        out = self.dropout(out)
        out = self.fc(out)
        return out

# 定义自定义数据集类
class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# 读取数据
data = pd.read_excel(r'F:\论文\05\01-返修1\2632只-2018-2022-划分后\医药生物_2018-2022.xlsx', header=0)
selected_cols = ['intLiab2Ev', 'netLiab2Ev', 'tbAsset', 'grsPtMrgnSq', 'opCostTtm', 'intLiab', 'fcfe', 'faTurnR', 'ev2Ebitda', 'opPtTtm', 'dNetPtYoy5', 'grsMrgnTtm', 'fairValChgTtm', 'eqtMrq', 'epsTtm', 'totalShare', 'bps', 'liqShareA', 'ncaAsset', 'ca2Asset', 'grsRevTtm', 'opPtYoy5', 'peTtmD', 'rdCost', 'assetLiabilityRatio', 'ebtYoy5', 'close']
data = data[selected_cols].values.astype('float32')


# 检测和处理 NaN 和无穷大值
data[np.isnan(data)] = 0  # 将 NaN 替换为 0
data[np.isinf(data)] = np.finfo(np.float32).max  # 将无穷大的值替换为 float32 的最大值

# 数据归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# 划分训练集和测试集
train_ratio = 0.8
train_size = int(len(data_scaled) * train_ratio)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# 创建训练集和测试集的数据加载器
batch_size = 16
train_dataset = StockDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = StockDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 设置设备为GPU或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 修改模型参数和超参数
input_size = len(selected_cols)  # 输入特征数
output_size = 1  # 输出大小（预测的收盘价）
num_epochs = 50
learning_rate = 0.0001

# 初始化模型
hidden_size = 64  # 隐藏层大小
num_layers = 2  # LSTM层数
dropout_rate = 0.3  # Dropout率
model = ATCNNLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)


# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data in train_loader:
        inputs = data[:, :-1].unsqueeze(2).to(device)
        targets = data[:, -1].unsqueeze(1).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_dataset)

    # 在测试集上进行评估
    model.eval()
    test_loss = 0.0
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            inputs = data[:, :-1].unsqueeze(2).to(device)
            targets = data[:, -1].unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            predictions.extend(outputs.squeeze(1).cpu().numpy())
    test_loss /= len(test_dataset)

    # 计算评估指标
    targets = test_dataset.data[:, -1]
    rmse = mean_squared_error(targets, predictions, squared=False)
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    # 打印训练过程中的指标
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, RMSE={rmse:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")