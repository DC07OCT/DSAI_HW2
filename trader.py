# You can write code above the if-main block.

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()


##我的程式碼
    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

    train_filepath = args.training
    test_filepath = args.testing
    train_data = pd.read_csv(train_filepath, names=['open', 'high', 'low', 'close'], header=None)
    test_data = pd.read_csv(test_filepath, names=['open', 'high', 'low', 'close'], header=None)
    print(train_data)
    print(test_data)

    #複製dataframe
    train_price = train_data.copy()  # deep=True複製
    test_price = test_data.copy()

    # data normalize(後面會再mapping回去)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # 把trainset和testset的值合併在一起,等等一起做normalized
    open_data = np.append(train_price['open'].values.reshape(-1, 1),test_price['open'].values.reshape(-1, 1))  # numpy array
    high_data = np.append(train_price['high'].values.reshape(-1, 1), test_price['high'].values.reshape(-1, 1))
    low_data = np.append(train_price['low'].values.reshape(-1, 1), test_price['low'].values.reshape(-1, 1))
    close_data = np.append(train_price['close'].values.reshape(-1, 1), test_price['close'].values.reshape(-1, 1))

    # 再trainset和testset的值合併一起做normalized 分四個scaler
    scaler_open = scaler.fit(open_data.reshape(-1, 1))
    scaler_high = scaler.fit(high_data.reshape(-1, 1))
    scaler_low = scaler.fit(low_data.reshape(-1, 1))
    scaler_close = scaler.fit(close_data.reshape(-1, 1))

    # 分別transform
    train_price['open'] = scaler_open.transform(train_price['open'].values.reshape(-1, 1))
    test_price['open'] = scaler_open.transform(test_price['open'].values.reshape(-1, 1))
    train_price['high'] = scaler_high.transform(train_price['high'].values.reshape(-1, 1))
    test_price['high'] = scaler_high.transform(test_price['high'].values.reshape(-1, 1))
    train_price['low'] = scaler_low.transform(train_price['low'].values.reshape(-1, 1))
    test_price['low'] = scaler_low.transform(test_price['low'].values.reshape(-1, 1))
    train_price['close'] = scaler_close.transform(train_price['close'].values.reshape(-1, 1))
    test_price['close'] = scaler_close.transform(test_price['close'].values.reshape(-1, 1))


    #print(train_price)
    #print(test_price)

    train_data_raw = train_price.to_numpy()  # convert to numpy array
    test_data_raw = test_price.to_numpy()  # convert to numpy array
    # 轉numpy後自動把dataframe中一row弄成一包就是['open','high','low','close']

    x_train = []#input data
    y_train = []#label data
    x_test = []#input data
    y_test = []#label data
    lookback = 20  # 用前19天預測第20天

    for index in range(len(train_data_raw) - lookback):
        x_train.append(train_data_raw[index:(index + lookback - 1)])
        # y_train.append(scaler_open.inverse_transform(train_data_raw[index + lookback-1][0].reshape(1,-1)))#也可以
        y_train.append(train_data['open'][index + lookback - 1])  # label用沒normalized的

    for index in range(len(test_data_raw) - 1):

        if index < (lookback - 1):
            if index == 0:
                x_test.append(train_data_raw[(index - lookback + 1):])
            else:
                temp = np.vstack((train_data_raw[(index - lookback + 1):], test_data_raw[:index]))
                x_test.append(temp)
        else:
            x_test.append(test_data_raw[index - lookback + 1:index])  # test data用到training data的後(lookback-1)筆
        y_test.append(test_data['open'][index])

    x_train = np.array(x_train)  # training set 的 input包(19天) 集合
    y_train = np.array(y_train)  # training set 的 output包(第20天) 集合
    x_test = np.array(x_test)  # testing set 的 input包(19天) 集合
    y_test = np.array(y_test)  # testing set 的 output包(第20天) 集合


    val_set_size = int(np.round(0.2 * x_train.shape[0]))#算20%資料量是幾筆
    train_set_size = x_train.shape[0] - val_set_size
    # 切割成train set(後80%) validation set(前20%)
    x_val = x_train[:val_set_size]
    y_val = y_train[:val_set_size]

    x_train = x_train[val_set_size:]
    y_train = y_train[val_set_size:]


    import torch
    import torch.nn as nn

    #data 轉成tensor準備放入model
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_val = torch.from_numpy(x_val).type(torch.Tensor)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)  # ground truth for lstm用
    y_val_lstm = torch.from_numpy(y_val).type(torch.Tensor)  # ground truth for lstm用
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    #print('here', x_train.shape)

    # parameters
    input_dim = 4  # 放入 open,high ,low,close 做training #input_dim是指輸入維度
    hidden_dim = 32  # 代表一層hidden layer有32個LSTM neuron
    num_layers = 2  # 2層hidden layer
    output_dim = 1  # 最後要predict的日子是幾天(predict 1 day 的 open price)
    num_epochs = 15000


    class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):  # 4,32,2,1
            super(LSTM, self).__init__()
            self.hidden_dim = hidden_dim  # hidden_dim=hidden_layer的output dim(也是1個hidden layer的 LSTM neuron個數)
            self.num_layers = num_layers  # 幾層hidden layer(不能太多!!)

            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)  # ouput_dim 是指最後要predict的日子是幾天(1!!)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])
            return out


    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    hist = np.zeros(num_epochs)  # 用來記錄歷史值

    # model.train()
    for t in range(num_epochs):
        y_train_pred = model(x_train.float())
        loss = criterion(y_train_pred.squeeze(1).float(), y_train_lstm.float()).requires_grad_()
        print("Epoch ", t, "MSE: ", loss.item())
        if loss.item() < 2:#提早跳出epoch
            break
        hist[t] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation
    from sklearn.metrics import mean_squared_error
    import math

    # make predictions for validation and test
    y_val_pred = model(x_val.float())
    y_test_pred = model(x_test.float())
    validationScore = math.sqrt(mean_squared_error(y_val[:], y_val_pred[:].detach().numpy()))
    print('Validation Score: %.2f RMSE' % (validationScore))
    testScore = math.sqrt(mean_squared_error(y_test[:], y_test_pred[:].detach().numpy()))
    print('Test Score: %.2f RMSE' % (testScore))


    def predict_action(now_state, today_open, inp):
        model.eval()
        tomorrow_pred = model(inp.float())  # 放入一筆資料
        if now_state == -1:
            if tomorrow_pred >= today_open:
                return 1, 0
            else:
                return 0, -1
        elif now_state == 1:
            if tomorrow_pred > today_open:
                return 0, 1
            else:
                return -1, 0
        else:
            if tomorrow_pred > today_open:
                return 1, 1
            elif tomorrow_pred == today_open:
                return 0, 0
            else:
                return -1, -1


    # 寫入csv file
    import csv
    now = 0
    with open('output2.csv', 'w', newline="") as output_file:
        writer = csv.writer(output_file)
        for row in range(len(x_test)):
            if (row - 1) < 0:  # 第一筆的前一天值要考慮到training set
                action, now = predict_action(now, y_train[row - 1],
                                             torch.index_select(x_test, 0, torch.LongTensor([row])))
            else:
                action, now = predict_action(now, y_test[row - 1],
                                             torch.index_select(x_test, 0, torch.LongTensor([row])))
            print(action)
            writer.writerow([str(action)])
