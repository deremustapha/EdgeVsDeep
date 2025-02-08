import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from time import time


from utils.data import *
from utils.preprocessing import *
from utils.models import *
from utils.evaluation import *
from vit_pytorch.cct import CCT
from mcunet.mcunet.model_zoo import net_id_list, build_model, download_tflite
from codecarbon import EmissionsTracker

fs = 2048
mains = 60.0
low_cut = 10.0
high_cut = 450.0 
order=5
train_percent = 80 # percentage
batch_size = 32
window_time = 32
overlap_percent = 60
no_channels = 256
epochs = 1
learning_rate = 0.001


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

def train_multiple_subjects(start_subject, stop_subject, sessions=[1,2], base_path='base', save_path='result.csv'):
    """
    Train models for multiple subjects and save results to CSV
    
    Args:
        start_subject (int): Starting subject number
        stop_subject (int): Ending subject number (inclusive)
        sessions (list): List of sessions to use
    """
    results = []
    
    for subject in range(start_subject, stop_subject + 1):
        print(f"\nProcessing Subject {subject}")
        
        # Load and preprocess data
        X, y = db3_load_per_subject(base_path, subject=subject, sessions=sessions)
        # print(f"X shape: {X.shape}")
        # print(f"y shape: {y.shape}")
        
        preprocess = EMGPreprocessing(fs=fs, notch_freq=mains, low_cut=low_cut, 
                                    high_cut=high_cut, order=order)
        X_notch = preprocess.remove_mains(X)
        X_band = preprocess.bandpass_filter(X_notch)
        
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

        ##################################

    

        model = MCUNet(number_gestures)

        # ##
        # input_channels = 256
        # conv_filters = 128
        # lstm_hidden_size = 128
        # lstm_layers = 3
        # model = CTRLEMG(input_channels, conv_filters, lstm_hidden_size, lstm_layers, number_gestures)
        
        #model = ProxylessNAS(number_gestures)
        
        # model = CCT(
        #     img_size = (256, 65),
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
        #              )
        #######

        #model = ResNet18(number_gestures)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"Training Subject {subject} on {device}")
        
        # Training
        train_accuracy_per_epoch = []
        tracker = EmissionsTracker()
        tracker.start()
        start_time = time()
        for epoch in tqdm(range(epochs), desc=f"Training Subject {subject}"):
            train_loss, train_acc = train_loop(model, device, train_dataloader, 
                                             criterion, optimizer)
            train_accuracy_per_epoch.append(train_acc)
        
        tracker.stop()
        end_time = time()  
        # Calculate metrics
        avg_train_acc = np.mean(train_accuracy_per_epoch)
        final_train_acc = train_accuracy_per_epoch[-1]
        test_loss, test_acc = test_loop(model, device, test_dataloader, criterion)
        
        # print(f"Subject {subject} Results:")
        # print(f"Average Train Accuracy: {avg_train_acc*100:.2f}%")
        # print(f"Final Train Accuracy: {final_train_acc*100:.2f}%")
        # print(f"Test Accuracy: {test_acc*100:.2f}%")
        
        # Store results
        results.append({
            'Subject': subject,
            'Avg_Train_Accuracy': avg_train_acc * 100,
            'Test_Accuracy': test_acc * 100
        })

        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        
        print(f'Training Time is {end_time - start_time}')
        print(f"Subject {subject} Results:")
        print(f"Average Train Accuracy: {avg_train_acc*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        # Free up memory
        del model
        torch.cuda.empty_cache()

    return df

if __name__ == "__main__":
    # Example usage:
    base_path = ""
    save_path = ''
    results = train_multiple_subjects(1, 1, sessions=[1,2], base_path=base_path, save_path=save_path)
    print("\nFinal Results:")
    print(results)