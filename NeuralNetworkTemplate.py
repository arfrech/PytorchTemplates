#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch, numpy, pandas
class NeuralNetwork(torch.nn.Module):
    
    def __init__(self,in_features,out_features,model_type='Classifier',*,epochs=150,hidden_size=[64,16,64],
                 dropout=0.0,optimizerAlgo='Adam',learning_rate=0.01,lambdaL2=0,batchsize=None,
                 shuffle=True,drop_last=True,doBN=False,verbose=True,random_state=None):
        '''
        in_features = int -> número de features do dataset
        out_features = int -> número de saídas do modelo. Para classificacão binária ou regressão, out_features = 1
        model_type = str -> indicar tipo do modelo, 'Classifier' ou 'Regressor'
        epochs = int -> 
        hidden_size = list -> dentro da lista, os números são a quantidade de neurônios e a posicão o layer, por exemplo hidden_size = [64,16,64]
        dropout = float entre 0 e 1 -> porcentagem dos neurônios que serão desconsiderados em cada layer, por exemplo dropout = 0.5 irá desconsiderar metade do layer
        optimizerAlgo = str -> nome do algoritmo de otimizacão do pytorch, por exemplo 'Adam', 'SGD'
        learning_rate = float -> learning rate do otimizador
        lambdaL2 = float ->
        batchsize = int -> indicar quantidade de exemplares por batch. Para não usar batch, batchsize = None
        shuffle = bool -> caso usar batch, definir shuffle ou não para a separacão dos exemplares nos batchs
        drop_last = bool -> caso usar batch, definir se o último bath será ou não desconsiderado
        doBN = bool -> indicar uso ou não do Batch Normalization
        verbose = bool -> indicar se deseja o print da perda (loss) a cada 50 epochs
        random_state = int -> definir ou não uso de semente de repricabilidade
        '''
        
        super().__init__()
        self.random_state = random_state
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.n_layers = len(hidden_size)
        self.dropout = dropout
        self.epochs = epochs
        self.doBN = doBN
        self.verbose = verbose
        self.model_type = model_type
        self.optimizerAlgo = optimizerAlgo
        self.lambdaL2 = lambdaL2
        self.batchsize = batchsize
        if batchsize == 1:
            self.doBN = False
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.model = torch.nn.ModuleDict()
        x = self.in_features
        for i in numpy.arange(self.n_layers):
            n_nodes = self.hidden_size[i]
            if i == 0:
                self.model['input'] = torch.nn.Linear(x,n_nodes)
            elif self.doBN:
                self.model[f'BatchNorm1d{i}'] = torch.nn.BatchNorm1d(x)
                self.model[f'hidden{i}'] = torch.nn.Linear(x,n_nodes)
            else:
                self.model[f'hidden{i}'] = torch.nn.Linear(x,n_nodes)
            x = n_nodes
        self.output = torch.nn.Linear(x,self.out_features)
        
    def forward(self,x):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
        for name,layer in self.model.items():
            if name == 'input':
                x = torch.nn.functional.relu(layer(x)).to(self.device) # x passa pelo input
            elif 'BatchNorm1d' in name:
                x = layer(x).to(self.device) # x passa pelo batch norm
            else:
                x = layer(x).to(self.device) # x passa pela hidden layer
                x = torch.nn.functional.relu(x).to(self.device) # x passa pela ReLU
                x = torch.nn.functional.dropout(x,p=self.dropout).to(self.device) # dropout na hidden layer
        
        output = self.output(x).to(self.device) # x passa pelo output se e termina
        return output
    
    def optimizer(self):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
        optifun = getattr(torch.optim,self.optimizerAlgo)
        optimizer = optifun(self.model.parameters(),lr=self.learning_rate,weight_decay=self.lambdaL2)
        return optimizer
    
    def criterion(self):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
        if self.model_type == 'Classifier':
            if self.out_features > 1:
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.BCEWithLogitsLoss()
        elif self.model_type == 'Regressor':
            criterion = torch.nn.HuberLoss()
        return criterion
    
    def fit(self,X_train,y_train):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
        
        if type(X_train) == pandas.core.frame.DataFrame or type(X_train) == pandas.core.series.Series:
            X_train_tensor = torch.FloatTensor(X_train.values)
        else:
            X_train_tensor = torch.FloatTensor(X_train)
        
        if type(y_train) == pandas.core.frame.DataFrame or type(y_train) == pandas.core.series.Series:
            if self.out_features == 1:
                y_train_tensor = torch.FloatTensor(y_train.values[:,None])
            else:
                y_train_tensor = torch.LongTensor(y_train.values)
        else:
            if self.out_features == 1:
                try:
                    y_train.shape[1]
                    y_train_tensor = torch.FloatTensor(y_train)
                except:
                    y_train_tensor = torch.FloatTensor(y_train[:,None])
                
            else:
                y_train_tensor = torch.LongTensor(y_train)
        
        if self.batchsize == None:
            self.batchsize = len(X_train)
        
        train_data = torch.utils.data.TensorDataset(X_train_tensor,y_train_tensor).to(self.device)
        train_loader = torch.utils.data.DataLoader(train_data,batch_size=self.batchsize,shuffle=self.shuffle,drop_last=self.drop_last)
        
        criterion = self.criterion()
        optimizer = self.optimizer()
        losses = []
        for epoch in range(self.epochs):
            self.model.train()
            batch_loss = []
            for X_train_i, y_train_i in train_loader:
                
                y_pred = self.forward(X_train_i)
                loss = criterion(y_pred,y_train_i)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.detach().numpy())
            
            losses.append(numpy.mean(batch_loss))
            if self.verbose == True:
                if epoch%50==0:
                    print(f'epoch {epoch} and loss is {loss}')
            
    def predict(self,X):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
            
        if type(X) != numpy.ndarray:
            X = X.values
        self.X = X
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i,data in enumerate(torch.FloatTensor(X).to(self.device)):
                y_val = self.forward(data.unsqueeze(0))
                if self.model_type == 'Classifier':
                    preds.append(y_val.argmax().item())
                elif self.model_type == 'Regressor':
                    preds.append(y_val.item())
        return numpy.array(preds)
    
    def metric(self,y_true):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
            
        if type(y_true) == torch.Tensor:
            y_true = y_true.detach().numpy()
            
        from sklearn.metrics import accuracy_score
        if self.model_type == 'Classifier':
            metric = accuracy_score(y_true,self.predict(self.X))
        elif self.model_type == 'Regressor':
            from sklearn.metrics import mean_absolute_percentage_error
            metric = mean_absolute_percentage_error(y_true,self.predict(self.X))
        
        return metric

class LSTM(torch.nn.Module):
    
    def __init__(self,input_size=4,output_size=1,*,epochs=50,hidden_size=128,fc_size=[128,64,16],
                 num_layers=1,dropout=0,optimizerAlgo='Adam',learning_rate=0.01,lambdaL2=0,
                 verbose=True,random_state=None):
        
        super().__init__()
        self.random_state = random_state
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        self.output_size = output_size 
        self.optimizerAlgo = optimizerAlgo
        self.learning_rate = learning_rate
        self.lambdaL2 = lambdaL2
        self.epochs = epochs
        self.num_layers = num_layers 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.fc_size = fc_size
        self.dropout = dropout
        self.verbose = verbose
        self.model = torch.nn.ModuleDict()
        self.model['lstm'] = torch.nn.LSTM(input_size=self.input_size,
                                           hidden_size=self.hidden_size,
                                           num_layers=self.num_layers,
                                           batch_first=True,
                                           bidirectional=False,
                                           dropout=self.dropout)
        x = self.hidden_size
        for i in numpy.arange(len(self.fc_size)):
            n_nodes = self.fc_size[i]
            self.model[f'fc_{i+1}'] = torch.nn.Linear(x,n_nodes)
            x = n_nodes
        self.output = torch.nn.Linear(x,self.output_size)
        
    def forward(self,x):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
        
        h_0 = torch.randn(self.num_layers,x.size(0),self.hidden_size).to(self.device) # hidden state
        c_0 = torch.randn(self.num_layers,x.size(0),self.hidden_size).to(self.device) # cell state
        
        for name,layer in self.model.items():
            if name == 'lstm':
                lstm_output,(hn,cn) = layer(x,(h_0,c_0))
                x = hn[-1].view(-1,self.hidden_size).to(self.device)
                x = torch.nn.functional.relu(x).to(self.device)
            else:
                x = torch.nn.functional.relu(layer(x)).to(self.device)
        output = self.output(x).to(self.device)
        return output
    
    def optimizer(self):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
            
        optifun = getattr(torch.optim,self.optimizerAlgo)
        optimizer = optifun(self.model.parameters(),lr=self.learning_rate,weight_decay=self.lambdaL2)
        return optimizer
    
    def criterion(self):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)

        criterion = torch.nn.MSELoss()
        return criterion
    
    def splitTimeSeries(self,X,y,window_size_split=1):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
        
        if type(X) == pandas.core.frame.DataFrame or type(X) == pandas.core.series.Series:
            X = X.values
        
        if type(y) == pandas.core.frame.DataFrame or type(y) == pandas.core.series.Series:
            y = y.values
        
        try:
            y.shape[1]
            y = y.flatten()
        except:
            pass
            
        time_series_data_x = []
        time_series_data_y = []
        time_series_len = len(y)
        for i in range(time_series_len-window_size_split):
            window = X[i:i+window_size_split]
            time_series_data_x.append(window)
            
            target = y[i+window_size_split:i+window_size_split+1]
            time_series_data_y.append(target)
        
        time_x_tensor = torch.FloatTensor(numpy.array(time_series_data_x))
        time_x_tensor = torch.reshape(time_x_tensor,(time_x_tensor.shape[0],window_size_split,time_x_tensor.shape[2]))

        time_y_tensor = torch.FloatTensor(numpy.array(time_series_data_y))
        
        return time_x_tensor.to(self.device), time_y_tensor.to(self.device)

    def fit(self,X_train,y_train,window_size_split=1):
        
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
        
        if type(X_train) != torch.Tensor or type(y_train) != torch.Tensor:
            X_train, y_train = self.splitTimeSeries(X=X_train,y=y_train,window_size_split=window_size_split)

        criterion = self.criterion()
        optimizer = self.optimizer()
        losses = []
        for epoch in range(self.epochs):
            self.model.train()
            y_pred = self.forward(X_train)
            loss = criterion(y_pred,y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().numpy())
            if self.verbose == True:
                if epoch%50==0:
                    print(f'epoch {epoch} and loss is {loss}')
                    
    def predict(self,X,y,window_size_split=1):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
            
        if type(X) != torch.Tensor or type(y) != torch.Tensor:
            X, y = self.splitTimeSeries(X=X,y=y,window_size_split=window_size_split)
        
        self.X = X
        self.y = y
        self.model.eval()
        with torch.no_grad():
            y_val = self.forward(X)
        
        return numpy.array(y_val)
    
    def metric(self):
        if type(self.random_state) == int:
            torch.manual_seed(self.random_state)
            
        from sklearn.metrics import mean_absolute_percentage_error
        metric = mean_absolute_percentage_error(self.y.detach().numpy(),self.predict(self.X,self.y))
        
        return metric
#End

