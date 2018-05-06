#Data: https://www.kaggle.com/c/titanic/data
#Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from fancyimpute import mice

#Functions

#Data Loading
def load(train_path, test_path):
    df_train_x = pd.read_csv(train_path)
    df_test_x = pd.read_csv(test_path)
    
    m_all_y = df_train_x.loc[:, 'Survived']
    df_train_x = df_train_x.drop('Survived', axis = 1)
    
    df_all_x = pd.concat([df_train_x, df_test_x], ignore_index = True)
    
    l_train = len(df_train_x)
    print(df_all_x.info())
    return df_all_x, l_train, m_all_y


#Write to file
def write(df_pred, l_train):
    m_temp = np.zeros((len(df_pred),2))
    for i in range(len(m_temp)):
        m_temp[i,0] = l_train + 1 + i
        m_temp[i,1]  = df_pred[i]
        
    df_sample = pd.DataFrame(m_temp,columns=['PassengerId', 'Survived']).astype(int)
    
    del m_temp, i
    
    df_sample.to_csv("F:/Code/Python/2 Titanic/submission.csv", index = False)
    return print("Write Completed")


#Data Wrangling
def wrangle(df_all_x, m_all_y):    
    #Sex
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by ='Survived', ascending = False)

    df_all_x['Sex'] = df_all_x['Sex'].map({'male' : 0, 'female' : 1})
    
    #Title
    df_all_x.loc[:,'Title'] = df_all_x['Name'].str.extract('(?<=, )(.*?)(?=\.)', expand = True).values
    df_all_x.loc[df_all_x['Title'].isin(['Mme', 'Mrs']), 'Title'] = '0'
    df_all_x.loc[df_all_x['Title'].isin(['Mlle', 'Ms', 'Miss']), 'Title'] = '1'
    df_all_x.loc[df_all_x['Title'].isin(['Don', 'Sir', 'Rev', 'Dr']) , 'Title'] = '5'
    df_all_x.loc[df_all_x['Title'].isin(['Dona', 'Jonkheer', 'Lady', 'the Countess']) , 'Title'] = '2'
    df_all_x.loc[df_all_x['Title'].isin(['Major', 'Capt', 'Col']) , 'Title'] = '4'
    df_all_x.loc[df_all_x['Title'].isin(['Master']), 'Title'] = '3'
    df_all_x.loc[df_all_x['Title'].isin(['Mr']), 'Title'] = '6'
    
    df_all_x['Title'] = df_all_x['Title'].astype(int)
    
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['Title', 'Survived']].groupby(['Title'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
    
    #Surname
    df_all_x.loc[:,'Surname'] = df_all_x['Name'].str.extract('(.*?)(?=\,)', expand = True).values
    df_all_x['SameFamily'] = df_all_x['Surname'].duplicated(keep = False).astype(int)
    
    #Tickets
    df_all_x['Ticket'] = (df_all_x['Ticket'].str.extract('.*?([0-9]*)$', expand = True))
    df_all_x.loc[df_all_x['Ticket'] == '', 'Ticket'] = '370160'
    df_all_x['Ticket'] = df_all_x['Ticket'].astype(int)

    df_all_x['TicketNumLength'] = df_all_x['Ticket'].apply(lambda x: len(str(x).split(' ')[-1])).astype(int)

    df_all_x['JointTicket'] = 1
    df_all_x.loc[df_all_x['Ticket'].duplicated(keep = False) == False, 'JointTicket'] = 0
    
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['JointTicket', 'Survived']].groupby(['JointTicket'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
    df_visual[['TicketNumLength', 'Survived']].groupby(['TicketNumLength'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
    
    #Fare
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    
    #Cabin
    df_all_x.loc[pd.isnull(df_all_x['Cabin']) == True, 'Cabin'] = 'X'
    df_all_x['Cabin'] = df_all_x['Cabin'].astype(str).str[0]
    df_all_x.loc[df_all_x['Cabin'] == "X", 'Cabin'] = '0'
    df_all_x.loc[df_all_x['Cabin'] == "D", 'Cabin'] = '1'
    df_all_x.loc[df_all_x['Cabin'] == "E", 'Cabin'] = '2'
    df_all_x.loc[df_all_x['Cabin'] == "B", 'Cabin'] = '3'
    df_all_x.loc[df_all_x['Cabin'] == "F", 'Cabin'] = '4'
    df_all_x.loc[df_all_x['Cabin'] == "C", 'Cabin'] = '5'
    df_all_x.loc[df_all_x['Cabin'] == "G", 'Cabin'] = '6'
    df_all_x.loc[df_all_x['Cabin'] == "A", 'Cabin'] = '7'
    df_all_x.loc[df_all_x['Cabin'] == "T", 'Cabin'] = '8'
    df_all_x['Cabin'] = df_all_x['Cabin'].astype(int)
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['Cabin', 'Survived']].groupby(['Cabin'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
    
    #Embarked
    df_all_x.loc[pd.isnull(df_all_x['Embarked']) == True, 'Embarked'] = 'S'
    df_all_x['Embarked'] = df_all_x['Embarked'].map({'C' : 0, 'Q' : 1, 'S' : 2})
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean().sort_values(by ='Survived', ascending=False)

    #Family size
    df_all_x['FamilySize'] = df_all_x['Parch'] + df_all_x['SibSp'] + 1
    
    #Dummifying Embarkation and Title
    #embarked_dummies = pd.get_dummies(df_all_x['Embarked']).astype(int)
    #title_dummies = pd.get_dummies(df_all_x['Title']).astype(int)
    #cabin_dummies = pd.get_dummies(df_all_x['Cabin']).astype(int)
    
    #Drop the Categorical
    df_all_x = df_all_x.drop('PassengerId', axis = 1)
    df_all_x = df_all_x.drop('Name', axis = 1)
    df_all_x = df_all_x.drop('Surname', axis = 1)
    #df_all_x = df_all_x.drop('Embarked', axis = 1)
    #df_all_x = df_all_x.drop('Title', axis = 1)
    #df_all_x = df_all_x.drop('Cabin', axis = 1)
    
    return df_all_x

def impute(df_all_x):
    df_all_x.columns
    df_filled_x = mice.MICE().complete(df_all_x.as_matrix())
    df_filled_x = pd.DataFrame(df_filled_x, columns = df_all_x.columns)
    df_filled_x['Age'] = np.round(df_filled_x['Age'])
    return df_filled_x

#New Features
def new_feat(df_all_x):
    df_all_x['ServentMisstress'] = 0
    df_all_x.loc[(df_all_x['JointTicket'] == 1) & (df_all_x['FamilySize'] == 1), 'ServentMisstress'] = 1
    df_all_x['FamilySize'] += df_all_x['ServentMisstress']
    
    df_all_x['FarePerPerson'] = df_all_x['Fare'] / df_all_x['FamilySize']
    
    df_all_x['ClassAge'] = df_all_x['Age'] * df_all_x['Pclass']
    df_all_x['ClassFare'] = df_all_x['Fare'] ** df_all_x['Pclass'].map({1 : 3, 2 : 2, 3 : 1})
    df_all_x['ClassFamily'] = df_all_x['FamilySize'] ** df_all_x['Pclass'].map({1 : 3, 2 : 2, 3 : 1})
    df_all_x['ClassSex'] = (df_all_x['Sex'] + 1) * df_all_x['Pclass'].map({1 : 3, 2 : 2, 3 : 1})
    df_all_x['ClassCabin'] = (df_all_x['Cabin'] + 1) * df_all_x['Pclass']
    df_all_x['ClassTitle'] = (df_all_x['Title'] + 1) * df_all_x['Pclass']        
    
    df_all_x['Child'] = 0
    df_all_x.loc[df_all_x['Age'] < 18,'Child'] = 1
    
    df_all_x['Mother'] = 0
    df_all_x.loc[(df_all_x['Sex'] == 1) & (df_all_x['Parch'] > 0 ) & (df_all_x['Age'] >= 18 ) & (df_all_x['Title'] != 1 ), 'Mother'] = 1
    
    df_all_x.loc[df_all_x['FamilySize'] == 4, 'FamilySize'] = '4'
    df_all_x.loc[df_all_x['FamilySize'] == 3, 'FamilySize'] = '3'
    df_all_x.loc[df_all_x['FamilySize'] == 2, 'FamilySize'] = '3'
    df_all_x.loc[df_all_x['FamilySize'] == 7, 'FamilySize'] = '2'
    df_all_x.loc[df_all_x['FamilySize'] == 1, 'FamilySize'] = '2'
    df_all_x.loc[df_all_x['FamilySize'] == 5, 'FamilySize'] = '1'
    df_all_x.loc[df_all_x['FamilySize'] == 6, 'FamilySize'] = '1'
    df_all_x.loc[df_all_x['FamilySize'] == 8, 'FamilySize'] = '0'
    df_all_x.loc[df_all_x['FamilySize'] == 11, 'FamilySize'] = '0'
    df_all_x['FamilySize'] = df_all_x['FamilySize'].astype(int)
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values(by ='Survived', ascending=False)

    return df_all_x


#End-training
def final(df_all_x, l_train, m_all_y):
    end_train_x = []
    end_train_y = []
    end_test_x = []
    end_train_x = pd.DataFrame(df_all_x[0:l_train].as_matrix()[:, 1:],
                               columns = df_all_x.columns[1:]
                               )
    end_train_y = m_all_y
    end_test_x = pd.DataFrame(df_all_x[l_train:].as_matrix()[:, 1:],
                              columns = df_all_x.columns[1:]
                              )

    return end_train_x, end_train_y, end_test_x


#Centering & Normalizing
def centernorm(end_train_x, end_test_x):
    end_means = end_train_x.mean(axis = 0)
    end_std = end_train_x.std(axis = 0)
    
    end_train_x -= end_means
    end_train_x /= end_std
    end_test_x -= end_means
    end_test_x /= end_std
    
    return end_train_x, end_test_x


#Drop the correlated Vars
def dropcor(end_train_x, end_test_x):
    print(pd.DataFrame(end_train_x).corr() > 0.6)
    end_train_x = end_train_x.drop('SibSp', axis = 1)
    end_test_x = end_test_x.drop('SibSp', axis = 1)
    
    end_train_x = end_train_x.drop('Parch', axis = 1)
    end_test_x = end_test_x.drop('Parch', axis = 1)

    end_train_x = end_train_x.drop('Fare', axis = 1)
    end_test_x = end_test_x.drop('Fare', axis = 1)

    end_train_x = end_train_x.drop('ClassFare', axis = 1)
    end_test_x = end_test_x.drop('ClassFare', axis = 1)

    end_train_x = end_train_x.drop('Sex', axis = 1)
    end_test_x = end_test_x.drop('Sex', axis = 1)

    end_train_x = end_train_x.drop('Cabin', axis = 1)
    end_test_x = end_test_x.drop('Cabin', axis = 1)

    end_train_x = end_train_x.drop('Title', axis = 1)
    end_test_x = end_test_x.drop('Title', axis = 1)
    
    return end_train_x, end_test_x


#Feature Selection
def feat_select(end_train_x, end_test_x)    :
    from sklearn.ensemble import RandomForestClassifier as rfc
    
    model_randfor = rfc(n_estimators = 1000,
                        max_features = None,
                        bootstrap = False,
                        oob_score = False,                                                
                        warm_start = True
                        )
    
    model_randfor.fit(end_train_x, end_train_y)
        
    importance = pd.DataFrame(model_randfor.feature_importances_,
                              columns = ['Importance'],
                              index = end_train_x.columns
                              )
    
    importance['Std'] = np.std([tree.feature_importances_
                                for tree in model_randfor.estimators_], axis = 0)
    
    importance = importance.sort_values(by = 'Importance',
                                        axis = 0,
                                        ascending = False
                                        )
        
    plt.bar(range(importance.shape[0]), 
            importance.loc[:, 'Importance'], 
            yerr = importance.loc[:, 'Std'], 
            align = 'center',
            
            )
    
    #Drop the Irrelevant Features
    end_train_x = end_train_x[importance.loc[importance['Importance'] > 0.00,].index]
    end_test_x = end_test_x[importance.loc[importance['Importance'] > 0.00,].index]
    
    return end_train_x, end_test_x, importance

#Script
np.random.seed(117)
df_all_x, l_train, m_all_y = load("F:/Code/Kaggle/Titanic/train.csv", "F:/Code/Kaggle/Titanic/test.csv")

df_all_x = wrangle(df_all_x, m_all_y)
df_all_x = impute(df_all_x)
df_all_x = new_feat(df_all_x)

end_train_x, end_train_y, end_test_x = final(df_all_x, l_train, m_all_y)
end_train_x, end_test_x = centernorm(end_train_x, end_test_x)

end_train_x, end_test_x = dropcor(end_train_x, end_test_x)
end_train_x, end_test_x, importance = feat_select(end_train_x, end_test_x)

#Deep Learning with pytorch
import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data_utils

tTrainX = Variable(torch.from_numpy(end_train_x.as_matrix()).type(torch.FloatTensor))
tTrainY = Variable(torch.from_numpy(end_train_y.as_matrix()).type(torch.LongTensor))
tTestX = Variable(torch.from_numpy(end_test_x.as_matrix()).type(torch.FloatTensor))

epochs = 50000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_hat = F.logsigmoid(self.fc3(x))
        return y_hat
    
net = Net()
print(net)

optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    preds = net(tTrainX)
    loss = loss_func(preds, tTrainY)
    optimizer.zero_grad()
    loss.backward()
    optimize0r.step()
    if epoch % 100 == 0:
        print('Epoch: %d - Loss: %.3f' % ((epoch + 1), loss))
        
correct = 0
total = 0
for record in tTestX:
    print(record)
    outputs = net(record)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))