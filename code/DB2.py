# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
from time import time

import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import random

from utils.data import *
from utils.preprocessing import *
from utils.models import *
from utils.evaluation import *
from vit_pytorch.cct import CCT

from mcunet.mcunet.model_zoo import net_id_list, build_model, download_tflite

# Parameters
fs = 200
mains = 60.0
low_cut = 5.0
high_cut = 99.0 
order = 5
train_percent = 80
batch_size = 32
window_time = 200
overlap_percent = 60
no_channels = 16
T_gestures = 7
epochs = 1
learning_rate = 0.001

base_path = "/mnt/d/AI-Workspace/sEMGClassification/EMBC_2025/dataset/Low/DB2_MyoArmbandDataset/PreTrain"

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)



def MCUNet(number_gestures):
    mcunet, _, _ = build_model(net_id="mcunet-in3", pretrained=False)
    mcunet.first_conv.conv = torch.nn.Conv2d(1, 16, kernel_size=(3, 3), 
                                            stride=(1, 1), padding=(1, 1), bias=True)
    mcunet.classifier = torch.nn.Linear(160, number_gestures)
    return mcunet


def train_multiple_subjects(start_subject, end_subject, base_path, T_gesture=7, male=True, save_path='results.csv'):
    """
    Train models for multiple subjects and store results in CSV
    """
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    for subject in range(start_subject, end_subject + 1):
        print(f"\nProcessing Subject {subject}")
        
        # Load and preprocess data
        X, y = db2_load_per_subject(path=base_path, subject=subject, T_gestures=7, 
                                  Truncate=992, male=male)
        
        preprocess = EMGPreprocessing(fs=fs, notch_freq=mains, low_cut=low_cut, 
                                    high_cut=high_cut, order=order)
        X_notch = preprocess.remove_mains(X)
        X_band = preprocess.bandpass_filter(X_notch)
        
        # Window the data
        data, target = window_with_overlap(data=X_band, label=y, window_time=window_time, 
                                         overlap=overlap_percent, no_channel=no_channels, fs=fs)
        data = np.expand_dims(data, axis=1)
        
        # Split and prepare data
        X_train, y_train, X_test, y_test = TSTS_spilt(data, target, train_percent)
        X_train, y_train = shuffle_data(data=X_train, labels=y_train)
        X_test, y_test = shuffle_data(data=X_test, labels=y_test)
        
        train_dataset = EMGDataset(data=X_train, label=y_train)
        test_dataset = EMGDataset(data=X_test, label=y_test)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        number_gestures = len(np.unique(y_train))


        ######################
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        ##########################
        #model = MCUNet(number_gestures)
        #model = model = ProxylessNAS(number_gestures)

        # input_channels = 8
        # conv_filters = 128
        # lstm_hidden_size = 128
        # lstm_layers = 3

        # model = CTRLEMG(input_channels, conv_filters, lstm_hidden_size, lstm_layers, number_gestures)

        # model = CCT(
        #     img_size = (8, 40),
        #     embedding_dim = 384,
        #     n_input_channels=1,
        #     n_conv_layers = 3,
        #     kernel_size = 3,
        #     stride = 3,
        #     padding = 1,
        #     activation= nn.ReLU,
        #     pooling_kernel_size = 2,
        #     pooling_stride = 2,
        #     pooling_padding = 1,
        #     num_layers = 2,
        #     num_heads = 3,
        #     mlp_ratio = 3.,
        #     num_classes = number_gestures,
        #     positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
        #     )


        model = ResNet18(number_gestures)
        model = model.to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        train_accuracy_per_epoch = []

        tracker = EmissionsTracker()
        tracker.start()
        start_time = time()

        for epoch in tqdm(range(epochs), desc=f"Training Subject {subject}"):
            train_loss, train_acc = train_loop(model, device, train_dataloader, 
                                             criterion, optimizer)
            train_accuracy_per_epoch.append(train_acc)
        
        # Calculate final metrics

        tracker.stop()
        end_time = time()

        avg_train_acc = np.mean(train_accuracy_per_epoch)
        test_loss, test_acc = test_loop(model, device, test_dataloader, criterion)
        
        # Store results
        results.append({
            'Subject': subject,
            'Avg_Train_Accuracy': avg_train_acc*100,
            'Test_Accuracy': test_acc*100
        })
        
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)

        print(f'The training time is {end_time - start_time}')
        print(f"Subject {subject} Results:")
        print(f"Average Train Accuracy: {avg_train_acc*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        # Free up memory
        del model
        torch.cuda.empty_cache()
    
    # Save results to CSV

    return df

if __name__ == "__main__":
    # Example usage:
    base_path = ""
    save_path = ''
    results = train_multiple_subjects(1, 1, base_path, male=True, save_path=save_path)
    print("\nFinal Results:")
    print(results)