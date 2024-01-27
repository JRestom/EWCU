import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np
import os
import subprocess
import requests
from evaluation import accuracy, eval_mia, simple_mia, compute_losses
from methods import unlearning_finetuning
from methods import unlearning_EWCU, unlearning_EWCU_2, unlearning_ts, blindspot_unlearner
import time
from helpers import count_frozen_parameters, aggregatedEFIM, EFIM, combine_loaders
import copy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())

# manual random seed is used for dataset partitioning
# to ensure reproducible results across runs
RNG = torch.Generator().manual_seed(42)

# download and pre-process CIFAR10
normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_set = torchvision.datasets.CIFAR10( 
    root="./data", train=True, download=True, transform=normalize
)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
# we split held out data into test and validation set
held_out = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=normalize
)

test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)

# download the forget and retain index split
local_path = "forget_idx.npy"
if not os.path.exists(local_path):
    response = requests.get(
        "https://storage.googleapis.com/unlearning-challenge/" + local_path
    )
    open(local_path, "wb").write(response.content)
forget_idx = np.load(local_path)

# construct indices of retain from those of the forget set
forget_mask = np.zeros(len(train_set.targets), dtype=bool)
forget_mask[forget_idx] = True
retain_idx = np.arange(forget_mask.size)[~forget_mask]

# split train set into a forget and a retain set
forget_set = torch.utils.data.Subset(train_set, forget_idx)
retain_set = torch.utils.data.Subset(train_set, retain_idx)

forget_loader = torch.utils.data.DataLoader(
    forget_set, batch_size=128, shuffle=True, num_workers=2
)
retain_loader = torch.utils.data.DataLoader(
    retain_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# download pre-trained weights
local_path = "weights_resnet18_cifar10.pth"
if not os.path.exists(local_path):
    response = requests.get(
        "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth"
    )
    open(local_path, "wb").write(response.content)

weights_pretrained = torch.load(local_path, map_location=DEVICE)

# load model with pre-trained weights
model = resnet18(weights=None, num_classes=10)
model.load_state_dict(weights_pretrained)
model.to(DEVICE)
model.eval();

# Orignal model
print('---------Original model----------')
print(f"Train set accuracy: {100.0 * accuracy(model, train_loader):0.1f}%")
print(f"Test set accuracy: {100.0 * accuracy(model, test_loader):0.1f}%")
eval_mia(model, train_loader, test_loader, forget_loader)
print('\n')

# fine_train = []
# fine_test = []
# fine_mia = []

# ewcu_train = []
# ewcu_test = []
# ewcu_mia = []

# ewcu2_train = []
# ewcu2_test = []
# ewcu2_mia = []

# Finetuned
print('---------Unlearned model Finetuned_----------')
model = resnet18(weights=None, num_classes=10)
model.load_state_dict(weights_pretrained)
model.to(DEVICE)

start_time = time.time()
model_1 = unlearning_finetuning(model, retain_loader, 5)
end_time = time.time()
elapsed_time = end_time - start_time

acc_train = accuracy(model_1, train_loader)
acc_test = accuracy(model_1, test_loader)
print(f"Elapsed time: {elapsed_time} seconds")
print(f"Train set accuracy: {100.0 * acc_train:0.1f}%")
print(f"Test set accuracy: {100.0 * acc_test:0.1f}%")
e = eval_mia(model_1, train_loader, test_loader, forget_loader)

# fine_train.append(acc_train)
# fine_test.append(acc_test)
# fine_mia.append(e)
print('\n')

#EWCU
print('---------Unlearned model EWCU1----------')
model = resnet18(weights=None, num_classes=10)
model.load_state_dict(weights_pretrained)
model.to(DEVICE)

start_time = time.time()
model_2 = unlearning_EWCU(model, retain_loader, forget_loader, 5)
end_time = time.time()
elapsed_time = end_time - start_time
num_frozen_parameters = count_frozen_parameters(model_2)

print(f"number of frozen parameters: {num_frozen_parameters}")
print(f"Elapsed time: {elapsed_time} seconds")
acc_train = accuracy(model_2, train_loader)
acc_test = accuracy(model_2, test_loader)
print(f"Elapsed time: {elapsed_time} seconds")
print(f"Train set accuracy: {100.0 * acc_train:0.1f}%")
print(f"Test set accuracy: {100.0 * acc_test:0.1f}%")
e = eval_mia(model_2, train_loader, test_loader, forget_loader)

# ewcu_train.append(acc_train)
# ewcu_test.append(acc_test)
# ewcu_mia.append(e)
print('\n')

# EWCU2
print('---------Unlearned model EWCU2----------')
model = resnet18(weights=None, num_classes=10)
model.load_state_dict(weights_pretrained)
model.to(DEVICE)

start_time = time.time()
model_3 = unlearning_EWCU_2(model, retain_loader, forget_loader, 10)
end_time = time.time()
elapsed_time = end_time - start_time
num_frozen_parameters = count_frozen_parameters(model_3)

print(f"number of frozen parameters: {num_frozen_parameters}")
print(f"Elapsed time: {elapsed_time} seconds")
acc_train = accuracy(model_3, train_loader)
acc_test = accuracy(model_3, test_loader)
print(f"Elapsed time: {elapsed_time} seconds")
print(f"Train set accuracy: {100.0 * acc_train:0.1f}%")
print(f"Test set accuracy: {100.0 * acc_test:0.1f}%")
e = eval_mia(model_3, train_loader, test_loader, forget_loader)

# ewcu2_train.append(acc_train)
# ewcu2_test.append(acc_test)
# ewcu2_mia.append(e)
print('\n')

#SCRUB
print('---------Unlearned model SCRUB----------')
model = resnet18(weights=None, num_classes=10)
model.load_state_dict(weights_pretrained)
model.to(DEVICE)
model_4 = unlearning_ts(model, retain_loader, forget_loader, test_loader, epochs=5)
print(f"Retain set accuracy: {100.0 * accuracy(model_4, retain_loader):0.1f}%")
print(f"Test set accuracy: {100.0 * accuracy(model_4, test_loader):0.1f}%")
eval_mia(model_4, train_loader, test_loader, forget_loader)

# Bad-T
print('---------Unlearned model Bad-T----------')
model = resnet18(weights=None, num_classes=10)
model.load_state_dict(weights_pretrained)
model.to(DEVICE)

unlearning_teacher = resnet18(weights=None, num_classes=10).to(DEVICE)
full_trained_teacher = copy.deepcopy(model).to(DEVICE)
combined_loader = combine_loaders(forget_loader, retain_loader)

model_5 = blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, combined_loader, epochs = 5,
                optimizer = 'adam', lr = 0.01, 
                device = 'cuda', KL_temperature = 1)
print(f"Retain set accuracy: {100.0 * accuracy(model_5, retain_loader):0.1f}%")
print(f"Test set accuracy: {100.0 * accuracy(model_5, test_loader):0.1f}%")
eval_mia(model_5, train_loader, test_loader, forget_loader)


# print('---------Finetuned-------------')
# print(f'train: {np.mean(fine_train)}')
# print(f'test: {np.mean(fine_test)}')
# print(f'mia: {np.mean(fine_mia)}')
# print('\n')

# print('---------EWCU-------------')
# print(f'train: {np.mean(ewcu_train)}')
# print(f'test: {np.mean(ewcu_test)}')
# print(f'mia: {np.mean(ewcu_mia)}')
# print('\n')

# print('---------EWCU2-------------')
# print(f'train: {np.mean(ewcu2_train)}')
# print(f'test: {np.mean(ewcu2_test)}')
# print(f'mia: {np.mean(ewcu2_mia)}')
# print('\n')
