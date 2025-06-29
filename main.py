from model import *
from metric import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import warnings
import pandas as pd
import numpy as np
import torch.nn as nn

warnings.filterwarnings("ignore")

batch_size = 1024
random_state = 123
DATA_SET = 'RPI1807'
protein_feature = ['P' + str(i) for i in range(1, 830)]
rna_feature = ['R' + str(i) for i in range(1, 553)]
col_names = ['label'] + protein_feature + rna_feature
data = pd.read_csv(f'data/{DATA_SET}/sample.txt', names=col_names, sep='\t')

protein_size = len(protein_feature)
rna_size = len(rna_feature)

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

for fold, (train_index, test_index) in enumerate(kf.split(data)):
    print(f"Fold {fold + 1}/{k_folds}")

    train_data, test_data = data.iloc[train_index], data.iloc[test_index]

    train_label = pd.DataFrame(train_data['label'])
    train_data = train_data.drop(columns=['label'])
    train_tensor_data = TensorDataset(torch.from_numpy(np.array(train_data)), torch.from_numpy(np.array(train_label)))
    train_loader = DataLoader(train_tensor_data, shuffle=True, batch_size=batch_size)

    test_label = pd.DataFrame(test_data['label'])
    test_data = test_data.drop(columns=['label'])
    test_tensor_data = TensorDataset(torch.from_numpy(np.array(test_data)), torch.from_numpy(np.array(test_label)))
    test_loader = DataLoader(test_tensor_data, batch_size=batch_size)

    model = DBENet_NPI(protein_size, rna_size)
    loss_func = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

    epochs = 75
    for epoch in range(epochs):
        total_loss_epoch = 0.0
        total_train_loss_epoch = 0.0
        total_test_loss_epoch = 0.0
        total_train_tmp = 0
        total_test_tmp = 0

        model.train()
        for index, (x, y) in enumerate(train_loader):
            x, y = x.float(), y.float()
            y_hat = model(x)

            optimizer.zero_grad()
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            total_train_tmp += 1

        train_loss = total_loss_epoch / total_train_tmp
        train_auc, train_rec, train_pre, train_f1, train_acc, train_spe, train_mcc = get_result(train_loader, model)

        model.eval()

        with torch.no_grad():
            true_labels = []
            pred_probs = []

            for index, (x, y) in enumerate(test_loader):
                x, y = x.float(), y.float()
                y_hat = model(x)
                test_loss = loss_func(y_hat, y)
                true_labels.extend(y.tolist())
                pred_probs.extend(y_hat.tolist())
                total_test_loss_epoch += test_loss.item()
                total_test_tmp += 1

        test_loss = total_test_loss_epoch / total_test_tmp
        test_auc, test_rec, test_pre, test_f1, test_acc, test_spe, test_mcc = get_result(test_loader, model)
