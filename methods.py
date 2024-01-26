import torch
import helpers
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def unlearning_finetuning(model, retain, forget, validation, epochs):
    
    epochs = epochs

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()

    for _ in range(epochs):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model

def unlearning_EWCU(model, retain, forget, epochs, threshold):
    epochs = epochs

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()

    print('Computing EFIM')
    efim = helpers.EFIM(model, forget)
    print('EFIM Computed')
    threshold = threshold
    parameters_to_freeze, _ = helpers.get_parameters_with_small_norm(efim, threshold)
    parameters_to_freeze = [param for param in parameters_to_freeze if param not in ['fc.weight', 'fc.bias']]

    helpers.freeze_parameters(model, parameters_to_freeze)


    for _ in range(epochs):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model

