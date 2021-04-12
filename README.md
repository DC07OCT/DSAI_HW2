# Stock Prediction using LSTM
## 如何執行
pip install -r requirements.txt  
change directory to the trader.py file  
python trader.py --training training.csv --testing testing.csv --output output.csv  
## requirements
  	python==3.8.5  
	numpy==1.19.2  
	pandas==1.1.5  
	scipy==1.5.2  
	scikit-learn==0.23.2  
	torch==1.7.1  

## 資料前處理
 DATA NORMALIZATION BY MinMaxScaler(feature_range=(-1, 1))  
 把TRAINING DATA 切成 TRAININGSET(80%)和 VALIDATION SET(20%)  
 每20天為一個單位Predict,包好前19天的[OPEN,HIGH,LOW,CLOSE] 作為MODLEL TRAINING INPUT單位,最後以第20天的OPEN值為LABEL  
 TESTING DATA 前19筆要抓trainingset的後19筆資料作為input  

## 建立lstm模型
參數
input_dim = 4 #放入 open,high ,low,close 做training #input_dim是指輸入維度  
hidden_dim = 32#代表一層hidden layer有32個LSTM neuron  
num_layers = 2#2層hidden layer  
output_dim = 1 #最後要predict的日子是幾天  
num_epochs = 15000#經過試驗得到的  
![image](https://github.com/DC07OCT/DSAI_HW2/blob/main/1.png)
模型  
## TRAINING MODEL PROCESS
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)  
criterion = torch.nn.MSELoss(reduction='mean')  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
hist = np.zeros(num_epochs)#用來記錄歷史值  

#model.train()  
for t in range(num_epochs):  
    y_train_pred = model(x_train.float())  
    loss = criterion(y_train_pred.squeeze(1).float(), y_train_lstm.float()).requires_grad_()  
    print(loss.shape)  
    print("Epoch ", t, "MSE: ", loss.item())  
    hist[t] = loss.item()  
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  

## VALIDATION/Test SET 結果
![image](https://github.com/DC07OCT/DSAI_HW2/blob/main/2.png)

## 如何判斷ACTION
*當價格不變時我prefer往狀態0變

分析所有可能情況  
1.now_state=-1,它的action只可能為0 or +1  
如果預測結果是漲價(或相同)就+1,反之+0  
3.now_state=1,它的action只可能為0 or -1  
如果預測結果是跌價(或相同)就-1,反之+0  
5.now_state=0,它的action可能為0 or -1 or 1  
如果預測結果是跌價就-1,相同+0,漲價+1  
