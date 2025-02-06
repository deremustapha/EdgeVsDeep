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
from vit_pytorch.mobile_vit import MobileViT
from mcunet.mcunet.model_zoo import net_id_list, build_model, download_tflite
from codecarbon import EmissionsTracker

# Global configurations
fs = 1000
mains = 60.0
low_cut = 10.0
high_cut = 450.0 
order = 5
train_percent = 80 # percentage

train_trials = [1, 2, 3]
test_trials = [4, 5]

batch_size = 32
window_time = 64
overlap_percent = 60
no_channels = 64
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

def train_multiple_subjects(start_subject, stop_subject, base_path, save_path='results.csv'):
    """
    Train models for multiple subjects and store results in CSV
    
    Args:
        start_subject (int): Starting subject number
        stop_subject (int): Ending subject number (inclusive)
        base_path (str): Path to dataset
        save_path (str): Path to save CSV results
    """
    results = []
    
    for subject in range(start_subject, stop_subject + 1):
        print(f"\nProcessing Subject {subject}")
        
        # Load and preprocess data
        X_train, y_train = load_db4_data_per_trial(base_path, subject, train_trials)
        X_test, y_test = load_db4_data_per_trial(base_path, subject, test_trials)
        
        preprocess = EMGPreprocessing(fs=fs, notch_freq=mains, low_cut=low_cut, 
                                    high_cut=high_cut, order=order)
        
        # Apply filters
        X_train = preprocess.remove_mains(X_train)
        X_test = preprocess.remove_mains(X_test)
        X_train = preprocess.bandpass_filter(X_train)
        X_test = preprocess.bandpass_filter(X_test)
        
        # Window the data
        data_train, target_train = window_with_overlap(data=X_train, label=y_train, 
                                                      window_time=window_time, 
                                                      overlap=overlap_percent, 
                                                      no_channel=no_channels, fs=fs)
        data_test, target_test = window_with_overlap(data=X_test, label=y_test, 
                                                    window_time=window_time, 
                                                    overlap=overlap_percent, 
                                                    no_channel=no_channels, fs=fs)
        
        # Add channel dimension
        data_train = np.expand_dims(data_train, axis=1)
        data_test = np.expand_dims(data_test, axis=1)
        
        # Shuffle data
        X_train, y_train = shuffle_data(data=data_train, labels=target_train)
        X_test, y_test = shuffle_data(data=data_test, labels=target_test)
        
        # Create dataloaders
        train_dataset = EMGDataset(data=X_train, label=y_train)
        test_dataset = EMGDataset(data=X_test, label=y_test)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        number_gestures = len(np.unique(y_train))

        ###

        random_seed = 42
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)


        # model = MCUNet(number_gestures)
        #model = ProxylessNAS(number_gestures)

        # input_channels = no_channels
        # conv_filters = 128
        # lstm_hidden_size = 128
        # lstm_layers = 3
        # model = CTRLEMG(input_channels, conv_filters, lstm_hidden_size, lstm_layers, number_gestures)

        # model = CCT(
        #             img_size = (64, 64),
        #             embedding_dim = 384,
        #             n_input_channels=1,
        #             n_conv_layers = 3,
        #             kernel_size = 3,
        #             stride = 3,
        #             padding = 1,
        #             activation= nn.ReLU,
        #             pooling_kernel_size = 2,
        #             pooling_stride = 2,
        #             pooling_padding = 1,
        #             num_layers = 2,
        #             num_heads = 3,
        #             mlp_ratio = 3.,
        #             num_classes = number_gestures,
        #             positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
        #             )
        model = ResNet18(number_gestures)

        ###
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Train model
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
        # Get results
        avg_train_acc = np.mean(train_accuracy_per_epoch)
        final_train_acc = train_accuracy_per_epoch[-1]
        test_loss, test_acc = test_loop(model, device, test_dataloader, criterion)
        
        # Store results
        results.append({
            'Subject': subject,
            'Average Train Accuracy': avg_train_acc * 100,
            'Test Accuracy': test_acc * 100
        })
        
        # Save results after each subject
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        
        print(f'Training Time is {end_time - start_time}')
        print(f"\nSubject {subject} Results:")
        print(f"Average Train Accuracy: {avg_train_acc*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        # Free up memory
        del model
        torch.cuda.empty_cache()
    
    return df

if __name__ == "__main__":
    base_path = ""
    save_path = ''
    results_df = train_multiple_subjects(1, 1, base_path, save_path=save_path)
    print("\nFinal Results:")
    print(results_df)