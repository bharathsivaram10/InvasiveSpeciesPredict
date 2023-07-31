import numpy as np
import pandas as pd   
from sklearn.model_selection import train_test_split
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import torch.nn.functional as F
import seaborn as sns
import torch.optim as optim
from tqdm import tqdm
import pickle

from sklearn.preprocessing import MinMaxScaler    
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader,WeightedRandomSampler
from dat_mod import ClassifierDataset,MulticlassNN


def abstractions(file):

    df = pd.read_csv(file)

    # Get rid of data with less than 5 observations
    b = df.loc[:,['id_number','commonname']]
    b = b.drop_duplicates()

    c = list(b.commonname)
    c_unq = list(set(b))

    for name in c:
        if len(b.loc[b['commonname']==name]) < 5:
            print(name)
            df = df[df.commonname != name]

    a = df.loc[:,['geo_feature','feature_type','buffer_size']]
    a = a.drop_duplicates()

    b = df.loc[:,['id_number','commonname']]
    b = b.drop_duplicates()

    b = list(b.commonname)
    b_unq = list(set(b))

    # Here we make class dict to go from # class to common name
    num2name = {}
    name2num = {}

    for i in range(len(b_unq)):
        num2name[i] = b_unq[i]
        name2num[b_unq[i]] = i


    for i in range(len(b)):
        b[i] = name2num[b[i]]

    a['abstractions'] = a['geo_feature'] + '-' + a['feature_type'] + '-' + a['buffer_size'].astype(str)
    
    points = df.id_number.unique()

    newcols = list(set(a['abstractions']))
    print("Number of features in abstraction vector:", len(newcols))

    data = np.zeros((len(points),len(newcols)))

    for i in range(len(points)):
        current = df.loc[df['id_number'] == points[i]]
        current.set_index('id_number', inplace = True)
        for index, row in current.iterrows():
            ab = row.geo_feature + '-' + row.feature_type + '-' + str(row.buffer_size)
            new_ind = newcols.index(ab)
            data[i,new_ind] += row.value

    abstract = pd.DataFrame(data)
    abstract.columns = newcols
    abstract.index = points
    abstract.sort_index(inplace=True)

    abstract = abstract.assign(commonname=b)

    return abstract,num2name

def get_ready(df):

    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]  

    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=69)#, random_state=69)

    # Split train into train-val
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval,random_state=69)#,random_state=69)#, stratify=y_trainval)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Turn everything into numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train,y_train,X_val,y_val,X_test,y_test

def class_dist(obj,count_dict):
    
    for i in obj:
        if i in count_dict:
            count_dict[i] += 1           
        else:
            print("Check classes.")
            
    return count_dict

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

def train(X_train,y_train,X_val,y_val,X_test,y_test,count_dict,num2name):


    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    target_list = []
    for _, t in train_dataset:
        target_list.append(t)
    
    target_list = torch.tensor(target_list)

    class_count = [i for i in class_dist(y_train,count_dict).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    # print(class_weights)

    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(weights=class_weights_all,num_samples=len(class_weights_all),replacement=True)

    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0007
    NUM_FEATURES = len(X_train[0])
    NUM_CLASSES = len(list(count_dict.keys()))


    print(NUM_CLASSES,NUM_FEATURES)

    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE)#,
                          #sampler=weighted_sampler)


    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = MulticlassNN(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.015)
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.03, momentum=0.9)#, weight_decay=0.0005)


    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(train_loader) - 1)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    accuracy_stats = {'train': [],"val": []}
    loss_stats = {'train': [],"val": []}

    loss_tracker = np.zeros(3)
    track = 0

    print("Begin training.")
    for e in tqdm(range(1, EPOCHS+1)):
        
        try:
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            model.train()

            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer.zero_grad()
                
                y_train_pred = model(X_train_batch)
                
                train_loss = loss(y_train_pred, y_train_batch)
                train_acc = multi_acc(y_train_pred, y_train_batch)
                
                train_loss.backward()
                optimizer.step()
                
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
            
            
        # VALIDATION    
            with torch.no_grad():
                
                val_epoch_loss = 0
                val_epoch_acc = 0
                
                model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    
                    y_val_pred = model(X_val_batch)
                                
                    val_loss = loss(y_val_pred, y_val_batch)
                    val_acc = multi_acc(y_val_pred, y_val_batch)
                    
                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss/len(train_loader))
            loss_stats['val'].append(val_epoch_loss/len(val_loader))
            accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

            # lr_scheduler.step()
            los = val_epoch_loss/len(val_loader)
            loss_tracker[track] = los
            track += 1

            if e % 3 == 0: 
                track = 0

            print(track)
        
            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

            if np.all(np.diff(loss_tracker) > 0):
                break
        except KeyboardInterrupt:
            break

    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
    plt.show()

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for X_batch, y_test_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            loss_curr = loss(y_val_pred, y_val_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            test_loss += loss_curr.item()
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    # print(type(y_pred_list),len(y_pred_list))
    # print(type(y_test),y_test.shape)

    correct = 0
    for i in range(len(y_pred_list)):
        if y_pred_list[i] == y_test[i]:
            correct += 1


    print(correct,'/',len(y_pred_list))
    print("Test Loss:",test_loss/len(test_loader))

    # confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=num2name, index=num2name)

    # sns.heatmap(confusion_matrix_df, annot=False)
    # plt.show()

    #class_report = classification_report(y_test, y_pred_list,output_dict=True)

    # num_lables = len(list(class_report.keys()))
    # precision = float(0)
    # recall = float(0)

    # for k in class_report.keys():
    #     precision += class_report[k]['precision']
    #     recall += class_report[k]['recall']

    #print(classification_report(y_test, y_pred_list))

    # print('Avg Precision:', precision/num_lables)
    # print('Avg Recall:',recall/num_lables )

def main():

    # file = 'feats_summer.csv'

    # abstract,num2name= abstractions(file)

    # fileObj = open('data2.obj', 'wb')
    # pickle.dump(abstract,fileObj)
    # pickle.dump(num2name,fileObj)
    # fileObj.close()

    with open('data2.obj', 'rb') as file:
      
        # Call load method to deserialze
        abstract = pickle.load(file)
        num2name = pickle.load(file)

    print(type(abstract),type(num2name))

    # ys = np.array(list(abstract.commonname))
    # unique, frequency = np.unique(ys, return_counts = True)
    # plt.bar(unique,frequency)
    # plt.title('Spring Observations')
    # plt.xlabel('Class')
    # plt.ylabel('Occurence')
    # plt.show()


    count_dict = {}
    for i in num2name.keys():
        count_dict[i] = 0

    X_train,y_train,X_val,y_val,X_test,y_test = get_ready(abstract)    

    # Train
    # lst = [y_train,y_val,y_test]

    # for i in lst:
    #     unique, frequency = np.unique(i, return_counts = True)
    #     plt.bar(unique,frequency)
    #     plt.show()

    train(X_train,y_train,X_val,y_val,X_test,y_test,count_dict,num2name)
    
    pass

if __name__ == "__main__":
    main()