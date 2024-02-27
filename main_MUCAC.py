import os
import time
import random
import glob
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import importlib
import helpers
from sklearn import linear_model, model_selection

import copy
import methods

class MultiLabelHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(MultiLabelHead, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for image_path in self.image_path_list:
            file_name = image_path.split('/')[-1]
            identity = int(identities[file_name])
            if identity >= train_index and identity < unseen_index:
                gender = int(label_map[file_name]["gender"])
                if gender == -1: gender = 0
                smiling = int(label_map[file_name]["smiling"])
                if smiling == -1: smiling = 0
                young = int(label_map[file_name]["young"])
                if young == -1: young = 0
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []
        for image_path in self.image_path_list:
            file_name = image_path.split('/')[-1]
            identity = int(identities[file_name])
            if identity < train_index:
                gender = int(label_map[file_name]["gender"])
                if gender == -1: gender = 0
                smiling = int(label_map[file_name]["smiling"])
                if smiling == -1: smiling = 0
                young = int(label_map[file_name]["young"])
                if young == -1: young = 0
                self.labels.append((gender, identity, smiling,  young))
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)

class ForgetDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for image_path in self.image_path_list:
            file_name = image_path.split('/')[-1]
            identity = int(identities[file_name])
            if identity >= train_index and identity < retain_index:
                gender = int(label_map[file_name]["gender"])
                if gender == -1: gender = 0
                smiling = int(label_map[file_name]["smiling"])
                if smiling == -1: smiling = 0
                young = int(label_map[file_name]["young"])
                if young == -1: young = 0
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)

class RetainDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for image_path in self.image_path_list:
            file_name = image_path.split('/')[-1]
            identity = int(identities[file_name])
            if identity < unseen_index and identity >= retain_index:
                gender = int(label_map[file_name]["gender"])
                if gender == -1: gender = 0
                smiling = int(label_map[file_name]["smiling"])
                if smiling == -1: smiling = 0
                young = int(label_map[file_name]["young"])
                if young == -1: young = 0
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)

class UnseenDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []
        for image_path in self.image_path_list:
            file_name = image_path.split('/')[-1]
            identity = int(identities[file_name])
            if identity < unseen_index:
                continue
            gender = int(label_map[file_name]["gender"])
            if gender == -1: gender = 0
            smiling = int(label_map[file_name]["smiling"])
            if smiling == -1: smiling = 0
            young = int(label_map[file_name]["young"])
            if young == -1: young = 0
            self.labels.append((gender, identity, smiling, young))
            self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)

def test(model, dataloader):
    criterion = nn.BCEWithLogitsLoss()
    start_time = time.time()
    print(f'[Test]')
    model.eval()

    with torch.no_grad():
        running_loss = 0.
        total_corrects_gender = 0
        total_corrects_smiling = 0
        total_corrects_young = 0

        for inputs, (gender, identity, smiling, young) in dataloader:
            inputs = inputs.cuda()

            # Convert labels to the same format as the outputs (float)
            labels = torch.stack((gender, smiling, young), dim=1).type(torch.FloatTensor).cuda()

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            loss = criterion(outputs, labels)

            # Calculate accuracy for each label
            threshold = 0.5
            preds = (probs > threshold).int()
            total_corrects_gender += torch.sum(preds[:, 0] == gender.cuda())
            total_corrects_smiling += torch.sum(preds[:, 1] == smiling.cuda())
            total_corrects_young += torch.sum(preds[:, 2] == young.cuda())

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(test_set)
        acc_gender = total_corrects_gender.item() / len(test_set)
        acc_smiling = total_corrects_smiling.item() / len(test_set)
        acc_young = total_corrects_young.item() / len(test_set)
        acc_avg = (acc_gender + acc_smiling + acc_young) / 3.0

        print('Loss: {:.4f} Time: {:.4f}s'.format(epoch_loss, time.time() - start_time))
        print('Average Accuracy: {:.2f} | Gender Acc: {:.2f} | Smiling Acc: {:.2f} | Young Acc: {:.2f}'.format(
            acc_avg, acc_gender, acc_smiling, acc_young
        ))

        return epoch_loss, (acc_avg, acc_gender, acc_smiling, acc_young)
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_kl_divergence(model1, model2, dataloader):
    model1.eval()
    model2.eval()
    total_kl_div = 0.0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            log_probs1 = F.log_softmax(model1(inputs), dim=1)
            probs2 = F.softmax(model2(inputs), dim=1)
            kl_div = F.kl_div(log_probs1, probs2, reduction='batchmean')
            total_kl_div += kl_div.item() * inputs.size(0)

    return total_kl_div / len(dataloader.dataset)

@torch.no_grad()
def evaluation(model, data_loader):
    model.eval()

    running_loss_gender = 0.
    running_corrects_gender = 0

    running_loss_smiling = 0.
    running_corrects_smiling = 0

    running_loss_young = 0.
    running_corrects_young = 0

    for inputs, (gender, identity, smiling, young) in data_loader:
        inputs = inputs.cuda()
        gender = gender.cuda()
        smiling = smiling.cuda()
        young = young.cuda()

        outputs = model(inputs)

        probs = torch.sigmoid(outputs)
        outputs_gender = probs[:, 0]
        outputs_smiling = probs[:, 1]
        outputs_young = probs[:, 2]

        preds_gender = (outputs_gender > 0.5).long()
        preds_smiling = (outputs_smiling > 0.5).long()
        preds_young = (outputs_young > 0.5).long()

        running_corrects_gender += torch.sum(preds_gender == gender)
        running_corrects_smiling += torch.sum(preds_smiling == smiling)
        running_corrects_young += torch.sum(preds_young == young)

    epoch_acc_gender = running_corrects_gender.item() / len(data_loader.dataset)
    epoch_acc_smiling = running_corrects_smiling.item() / len(data_loader.dataset)
    epoch_acc_young = running_corrects_young.item() / len(data_loader.dataset)

    avg_accuracy = (epoch_acc_gender + epoch_acc_smiling + epoch_acc_young) / 3

    return {
        'Average Acc': avg_accuracy,
        'Gender Acc': epoch_acc_gender,
        'Smiling Acc': epoch_acc_smiling,
        'Young Acc': epoch_acc_young,
    }

def compute_losses(net, loader):
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    all_losses = []

    for inputs, (gender, identity, smiling, young) in loader:
        labels = torch.stack((gender, smiling,  young), dim=1).type(torch.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda()

        logits = net(inputs)

        losses = criterion(logits, labels).mean(dim=1).cpu().detach().numpy()
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def cal_mia(model):
    set_seed(42)

    forget_losses = compute_losses(model, forget_dataloader_test)
    unseen_losses = compute_losses(model, unseen_dataloader)

    np.random.shuffle(forget_losses)
    forget_losses = forget_losses[: len(unseen_losses)]

    samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(unseen_losses) + [1] * len(forget_losses)

    mia_scores = simple_mia(samples_mia, labels_mia)
    forgetting_score = abs(0.5 - mia_scores.mean())

    return {'MIA': mia_scores.mean(), 'Forgeting Score': forgetting_score}

def train(model, dataloader, epoch,  optimizer, criterion):
    set_seed(42)

    model.train()
    start_time = time.time()
    print(f'[Epoch: {epoch + 1} - Training]')

    running_loss = 0.
    running_corrects = 0
    running_corrects_gender = 0
    running_corrects_smiling = 0
    running_corrects_young = 0

    print_every = 20
    batch_losses = []

    # Load a batch data of images.
    for i, (inputs, (gender, identity, smiling, young)) in enumerate(dataloader):
        inputs = inputs.cuda()

        # Convert labels to the same format as the outputs (float).
        labels = torch.stack((gender, smiling, young), dim=1).type(torch.FloatTensor).cuda()

        # Forward inputs and get output.
        optimizer.zero_grad()
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)

        loss = criterion(outputs, labels)
        batch_losses.append(loss.item())

        # Get loss value and update the network weights.
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy for the current batch.
        threshold = 0.5
        preds = (probs > threshold).int()
        corrects = torch.sum(preds == labels.int()).item()
        running_corrects += corrects

        # Calculate accuracy for each label in the current batch.
        corrects_gender = torch.sum(preds[:, 0] == gender.int().to(device)).item()
        corrects_smiling = torch.sum(preds[:, 1] == smiling.int().to(device)).item()
        corrects_young = torch.sum(preds[:, 2] == young.int().to(device)).item()
        running_corrects_gender += corrects_gender
        running_corrects_smiling += corrects_smiling
        running_corrects_young += corrects_young

        # Print average loss for the last 'print_every' batches.
        if (i + 1) % print_every == 0:
            avg_loss = sum(batch_losses) / print_every
            print(f"[Batch {i+1}] Avg. Loss: {avg_loss:.4f}")
            batch_losses = []

    # Calculate average training accuracy.
    acc_avg = running_corrects / (len(dataloader.dataset) * 3)

    # Calculate average accuracy for each label.
    acc_gender = running_corrects_gender / len(dataloader.dataset)
    acc_smiling = running_corrects_smiling / len(dataloader.dataset)
    acc_young = running_corrects_young / len(dataloader.dataset)

    loss = running_loss / len(train_set)
    print('[Train] Loss: {:.4f}, Average Training Accuracy: {:.2f}, '
          'Gender Acc: {:.2f}, Smiling Acc: {:.2f}, Young Acc: {:.2f}, Time: {:.4f}s'.format(
        loss, acc_avg, acc_gender, acc_smiling, acc_young, time.time() - start_time))

    return loss, (acc_avg, acc_gender, acc_smiling, acc_young)

def evaluate_unlearn(model_to_eval, test_dataloader, unseen_dataloader, forget_dataloader_test, base_model_path):
    
    model_to_eval.eval()

    base_model = models.resnet18(pretrained=False)
    num_features = base_model.fc.in_features
    base_model.fc = MultiLabelHead(num_features, 3)
    base_model.load_state_dict(torch.load(base_model_path))
    base_model = base_model.cuda()

    test_acc = evaluation(model_to_eval, test_dataloader)
    unseen_acc = evaluation(model_to_eval, unseen_dataloader)
    mia = cal_mia(model_to_eval.cuda())
    kl = compute_kl_divergence(base_model, model_to_eval, forget_dataloader_test)

    print(f'Test Acc: {test_acc}')
    print(f'Unseen Acc: {unseen_acc}')
    print(f'MIA: {mia}')
    print(f'KL-Div w.r.t Retrained model: {kl}\n')

def load_unlearned_model(weights_path):
    unlearned_model = models.resnet18(pretrained=True)
    num_features = unlearned_model.fc.in_features
    unlearned_model.fc = MultiLabelHead(num_features, 3)
    unlearned_model = unlearned_model.cuda()
    # model_path = f'last_checkpoint_epoch_{num_original_epochs}_multi_label.pth'
    model_path = weights_path + 'pre_trained_last_checkpoint_epoch_8.pth'
    unlearned_model.load_state_dict(torch.load(model_path))
    return unlearned_model
    
def finetuning():   
    ####### Finetuning 
    print('Finetuning')
    unlearned_model = load_unlearned_model(weights_path)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)
    set_seed(42)
    num_epochs = 2
    for epoch in range(num_epochs):
        train(unlearned_model, retain_dataloader_train, epoch, optimizer, criterion)
        test(unlearned_model, test_dataloader)

    print('\nEvaluating Finetuning')
    evaluate_unlearn(unlearned_model, test_dataloader, unseen_dataloader, forget_dataloader_test, base_model_path  = weights_path + 'retrained_model_weights.pth')

def cf_k():
    ####### CF-k (Class-wise Forgetting)
    print('CF-K')
    unlearned_model = load_unlearned_model(weights_path)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, unlearned_model.parameters()), lr=0.001)
    set_seed(42)

    # Freeze all the parameters
    for param in unlearned_model.parameters():
        param.requires_grad = False

    # Only unfreeze the last three layers for fine-tuning
    for param in unlearned_model.layer3.parameters():
        param.requires_grad = True
    for param in unlearned_model.layer4.parameters():
        param.requires_grad = True
    for param in unlearned_model.avgpool.parameters():
        param.requires_grad = True
    for param in unlearned_model.fc.parameters():
        param.requires_grad = True

    num_epochs = 2
    for epoch in range(num_epochs):
        train(unlearned_model, retain_dataloader_train, epoch, optimizer, criterion)
        test(unlearned_model, test_dataloader)

    print('\nEvaluating CF-K')
    evaluate_unlearn(unlearned_model, test_dataloader, unseen_dataloader, forget_dataloader_test,  base_model_path  = weights_path + 'retrained_model_weights.pth')

def negative_grad():
    print('Negative Gradient')
    ####### Negative Gradient Ascent
    unlearned_model = load_unlearned_model(weights_path)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)
    set_seed(42)

    num_epochs = 2
    print_every = 20
    for epoch in range(num_epochs):
        running_loss = 0
        # Training on forget data with Gradient Ascent
        for batch_idx, (x_forget, y_forget_data) in enumerate(forget_dataloader_train):
            if x_forget.size(0) != 64:
                continue

            y_forget = torch.stack((y_forget_data[0], y_forget_data[2], y_forget_data[3]), dim=1).type(torch.FloatTensor).cuda()
            outputs_forget = unlearned_model(x_forget.cuda())

            # Gradient Ascent loss for forget data
            loss_ascent_forget = -criterion(outputs_forget, y_forget)

            optimizer.zero_grad()
            loss_ascent_forget.backward()
            optimizer.step()
            running_loss -= loss_ascent_forget.item() * x_forget.size(0)

            if (batch_idx + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(forget_dataloader_train)}] - Batch Loss: {loss_ascent_forget.item():.4f}")

        average_epoch_loss = running_loss / (len(forget_dataloader_train) * x_forget.size(0))
        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {average_epoch_loss:.4f}")

    print('\nEvaluating Negative Gradient')
    evaluate_unlearn(unlearned_model, test_dataloader, unseen_dataloader, forget_dataloader_test,   base_model_path  = weights_path + 'retrained_model_weights.pth')

def advanced_neg_grad():
    ####### Advanced NegGrad with Classification Loss
    print('Advanced Negative Gradient')
    unlearned_model = load_unlearned_model(weights_path)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)
    set_seed(42)

    num_epochs = 2
    print_every = 20
    for epoch in range(num_epochs):
        running_loss = 0
        # Training on forget data with Gradient Ascent
        for batch_idx, ((x_forget, y_forget_data), (x_retain, y_retain_data)) in enumerate(zip(forget_dataloader_train, retain_dataloader_train)):
            if x_forget.size(0) != 64 or x_retain.size(0) != 64:
                continue

            y_forget = torch.stack((y_forget_data[0], y_forget_data[2], y_forget_data[3]), dim=1).type(torch.FloatTensor).cuda()
            y_retain = torch.stack((y_retain_data[0], y_retain_data[2], y_retain_data[3]), dim=1).type(torch.FloatTensor).cuda()

            outputs_forget = unlearned_model(x_forget.cuda())
            outputs_retain = unlearned_model(x_retain.cuda())

            # Gradient Ascent loss for forget data
            loss_ascent_forget = -criterion(outputs_forget, y_forget)
            loss_retain = criterion(outputs_retain, y_retain)
            overall_loss = loss_ascent_forget + loss_retain

            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()
            running_loss += overall_loss.item() * x_forget.size(0)

            if (batch_idx + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(forget_dataloader_train)}] - Batch Loss: {loss_ascent_forget.item():.4f}")

        average_epoch_loss = running_loss / (len(forget_dataloader_train) * x_forget.size(0))
        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {average_epoch_loss:.4f}")

    print('\nEvaluating Advanced Negative Gradient')
    evaluate_unlearn(unlearned_model, test_dataloader, unseen_dataloader, forget_dataloader_test,  base_model_path  = weights_path + 'retrained_model_weights.pth')

def unsir():
    ####### UNSIR

    class Noise(nn.Module):
        def __init__(self, batch_size, *dim):
            super().__init__()
            self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

        def forward(self):
            return self.noise

    def float_to_uint8(img_float):
        """Convert a floating point image in the range [0,1] to uint8 image in the range [0,255]."""
        img_uint8 = (img_float * 255).astype(np.uint8)
        return img_uint8
    
    print('UNSIR')
    unlearned_model = load_unlearned_model(weights_path)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)

    set_seed(42)

    train_epoch_losses = []
    num_epochs = 2
    print_interval = 20
    for epoch in range(num_epochs):
        running_loss = 0

        for batch_idx, ((x_retain, y_retain_data), (x_forget, y_forget_data)) in enumerate(zip(retain_dataloader_train, forget_dataloader_train)):
            # Instead of converting y_retain to LongTensor, convert it to float
            y_retain = torch.stack((y_retain_data[0], y_retain_data[2], y_retain_data[3]), dim=1).type(torch.FloatTensor).cuda()

            batch_size_forget = y_forget_data[0].size(0)

            if x_retain.size(0) != 64 or x_forget.size(0) != 64:
                continue

            # initialize the noise
            noise_dim = x_retain.size(1), x_retain.size(2), x_retain.size(3)
            noise = Noise(batch_size_forget, *noise_dim).cuda() # [64, r, g, b]
            noise_optimizer = torch.optim.Adam(noise.parameters(), lr=0.01)
            noise_tensor = noise()[:batch_size_forget]

            # update the noise
            for _ in range(5):
                outputs = unlearned_model(noise_tensor)
                with torch.no_grad():
                    target_logits = unlearned_model(x_forget.cuda())
                # Minimize the similarity between perturbed_retain and forget features
                loss_noise = -F.mse_loss(outputs, target_logits)
                noise_optimizer.zero_grad()
                loss_noise.backward(retain_graph=True)
                noise_optimizer.step()

            # Train the model with nose and retain image
            noise_tensor = torch.clamp(noise_tensor, 0, 1).detach().cuda()
            outputs_1 = unlearned_model(noise_tensor)
            loss_1 = criterion(outputs_1, y_retain)

            outputs_2 = unlearned_model(x_retain.cuda())
            loss_2 = criterion(outputs_2, y_retain)

            joint_loss = loss_1 + loss_2

            optimizer.zero_grad()
            joint_loss.backward()
            optimizer.step()
            running_loss += joint_loss.item() * x_retain.size(0)

            original_image = x_retain[0].cpu().numpy().transpose(1, 2, 0)
            image1 = TF.to_pil_image(float_to_uint8(original_image))
            image2 = TF.to_pil_image(noise.noise[0].cpu())


            if batch_idx % print_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_dataloader_train)}] - Batch Loss: {joint_loss.item():.4f}")

        average_train_loss = running_loss / (len(retain_dataloader_train) * x_retain.size(0))
        train_epoch_losses.append(average_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {average_train_loss:.4f}")

        ### stage 2 repair 

        num_epochs = 1
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            train(unlearned_model, retain_dataloader_train, epoch, optimizer, criterion)
            test(unlearned_model, test_dataloader)

        print('\nEvaluating UNSIR')
        evaluate_unlearn(unlearned_model, test_dataloader, unseen_dataloader, forget_dataloader_test,   base_model_path  = weights_path + 'retrained_model_weights.pth')

def scrub():  

    class DistillKL(nn.Module):
        def __init__(self, T):
            super(DistillKL, self).__init__()
            self.T = T

        def forward(self, y_s, y_t):
            p_s = F.log_softmax(y_s/self.T, dim=1)
            p_t = F.softmax(y_t/self.T, dim=1)
            loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
            return loss
        
    class SCRUBTraining:

        def __init__(self, teacher, student, retain_dataloader, forget_dataloader):
            self.teacher = teacher
            self.student = student
            self.retain_dataloader = retain_dataloader
            self.forget_dataloader = forget_dataloader

            self.criterion_cls = nn.BCEWithLogitsLoss()
            self.criterion_div = DistillKL(4.0)
            self.optimizer = optim.SGD(student.parameters(), lr=0.001)

        def train_epoch(self, print_every=20):
            set_seed(42)
            self.student.train()
            self.teacher.eval()

            running_loss = 0.
            running_corrects = 0
            batch_losses = []

            # Training with retain data
            for i, (inputs, (gender, identity, smiling, young)) in enumerate(self.retain_dataloader):
                inputs = inputs.cuda()
                labels = torch.stack((gender, smiling, young), dim=1).type(torch.FloatTensor).cuda()

                # Forward pass: Student (remove torch.no_grad() block for student)
                self.optimizer.zero_grad()  # Reset gradients to zero before computation
                outputs_student = self.student(inputs)
                probs_student =  torch.sigmoid(outputs_student)

                # Forward pass: Teacher
                with torch.no_grad():
                    outputs_teacher = self.teacher(inputs)

                # Compute classification loss
                loss_cls = self.criterion_cls(outputs_student, labels)
                batch_losses.append(loss_cls.item())

                # Compute divergence loss with teacher's outputs
                loss_div_retain = self.criterion_div(outputs_student, outputs_teacher)
                loss = loss_cls + loss_div_retain

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                threshold = 0.5
                preds = (probs_student > threshold).int()
                corrects = torch.sum(preds == labels.int()).item()
                running_corrects += corrects

                # Print average loss for the last 'print_every' batches
                if (i + 1) % print_every == 0:
                    avg_loss = sum(batch_losses) / print_every
                    print(f"[Batch {i+1}] Avg. Loss: {avg_loss:.4f}")
                    batch_losses = []

            # Calculate average loss and accuracy
            avg_loss = running_loss / len(self.retain_dataloader.dataset)
            avg_accuracy = running_corrects / (len(self.retain_dataloader.dataset) * 3) * 100
            print(f'[Epoch: {epoch + 1}] Loss: {avg_loss:.4f}, Average Training Accuracy: {avg_accuracy:.2f}%')

            return avg_loss, avg_accuracy
    
    print('SCRUB')
    original_model = load_unlearned_model(weights_path)
    scrub_model = load_unlearned_model(weights_path)

    set_seed(42)
    teacher = original_model
    student = scrub_model

    # Initialize and train
    scrub_trainer = SCRUBTraining(teacher, student, retain_dataloader_train, forget_dataloader_train)

    num_epochs = 2
    for epoch in range(num_epochs):
        scrub_trainer.train_epoch()
        print(f"Epoch {epoch+1} completed.")

    print('\nEvaluating SCRUB')
    evaluate_unlearn(student, test_dataloader, unseen_dataloader, forget_dataloader_test,  base_model_path  = weights_path + 'retrained_model_weights.pth')

def ewcu():
    ####### EWCU
    
    print('EWCU')
    def unlearning_EWCU(model, retain, forget, epochs, threshold=0.05):
        
        model.train()
        epochs = epochs
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        

        print('Computing EFIM')
        efim = helpers.EFIM(model, forget)
        
        print('EFIM Computed')
        agg_efim = helpers.aggregatedEFIM(efim)
        print(agg_efim)
        print(len(agg_efim))
        
        parameters_to_freeze = helpers.params_below_threshold(agg_efim, threshold)
        parameters_to_freeze = [param for param in parameters_to_freeze if param not in ['fc.weight', 'fc.bias','fc.fc.bias', 'fc.fc.weight']]
        print(parameters_to_freeze)
        print(len(parameters_to_freeze))

        helpers.freeze_parameters(model, parameters_to_freeze)
        

        for _ in range(epochs):
            for i, (inputs, (gender, identity, smiling, young)) in enumerate(retain):
                inputs = inputs.cuda()
                # Convert labels to the same format as the outputs (float).
                labels = torch.stack((gender, smiling, young), dim=1).type(torch.FloatTensor).cuda()

                # Forward inputs and get output.
                optimizer.zero_grad()
                outputs = model(inputs)
                #probs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                

                # Get loss value and update the network weights.
                loss.backward()
                optimizer.step()

                #running_loss += loss.item() * inputs.size(0)
                
            scheduler.step()
            
        model.eval()
        return model

    unlearned_model = load_unlearned_model(weights_path)

    frozen_params = sum(1 for _, param in unlearned_model.named_parameters() if not param.requires_grad)
    print(f"Number of frozen parameters: {frozen_params}")

    num_named_parameters = sum(1 for _ in unlearned_model.named_parameters())
    print(f"Number of named parameters in the model: {num_named_parameters}")

    importlib.reload(helpers)
    unlearned_model = unlearning_EWCU(unlearned_model, retain_dataloader_train, forget_dataloader_train, 2)
    
    print('\nEvaluating EWCU')
    evaluate_unlearn(unlearned_model, test_dataloader, unseen_dataloader, forget_dataloader_test,  base_model_path  = weights_path + 'retrained_model_weights.pth')


def bad_teacher():
    ####### Bad-T
    print('Bad-T')
    unlearned_model = load_unlearned_model(weights_path)

    unlearning_teacher = models.resnet18(pretrained=False)
    num_features = unlearning_teacher.fc.in_features
    unlearning_teacher.fc = MultiLabelHead(num_features, 3)
    unlearning_teacher = unlearning_teacher.to(device)
    full_trained_teacher = copy.deepcopy(unlearned_model).to(device)
    combined_loader = helpers.combine_loaders(forget_dataloader_test, retain_dataloader_train)

    unlearned_model = methods.blindspot_unlearner(unlearned_model, unlearning_teacher, full_trained_teacher, combined_loader, epochs = 2,
    optimizer = 'adam', lr = 0.01, 
    device = 'cuda', KL_temperature = 1)

    print('\nEvaluating Bad-T')
    evaluate_unlearn(unlearned_model, test_dataloader, unseen_dataloader, forget_dataloader_test,  base_model_path  = weights_path + 'retrained_model_weights.pth')

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/share/users/sara.pieri/EWCU/data/"
weights_path = "/share/users/sara.pieri/EWCU/weights/"
source_root = data_path + 'CelebAMask-HQ/CelebA-HQ-img/'
attribute_path = data_path + 'CelebA-HQ-attribute.txt'

identities = {}
with open(data_path + 'CelebA-HQ-identity.txt') as f:
    lines = f.readlines()
    for line in lines:
        file_name, identity = line.strip().split()
        identities[file_name] = identity

attributes_map = {
    "gender": 21,
    "smiling": 32,
    "young": 40
}

label_map = {}
with open(attribute_path) as f:
    lines = f.readlines()
    for line in lines[2:]:
        splited = line.strip().split()
        file_name = splited[0]
        label_map[file_name] = {attr: int(splited[idx]) for attr, idx in attributes_map.items()}

# There are 6217 identities. There are 30000 images. There are 30000 images.
# Overlaps {'train_test': 0, 'train_forget': 10183, 'train_retain': 15750, 'train_unseen': 0, 'test_forget': 0, 'test_retain': 0, 'test_unseen': 0, 'forget_retain': 0, 'forget_unseen': 0, 'retain_unseen': 0}
train_index = 190
retain_index = 1250
unseen_index = 4855

train_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor()
])

unseen_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor()
])

train_set = TrainDataset(transform=train_transform)
test_set = TestDataset(transform=test_transform)
forget_set_train = ForgetDataset(transform=train_transform)
forget_set_test = ForgetDataset(transform=test_transform)
retain_set_train = RetainDataset(transform=train_transform)
retain_set_test = RetainDataset(transform=test_transform)
unseen_set = UnseenDataset(transform=unseen_transform)

print('Data statistics:')
print('\tTrain dataset size:', len(train_set))
print('\tTest dataset size:', len(test_set))
print('\tForget dataset size:', len(forget_set_train))
print('\tRetain dataset size:', len(retain_set_train))
print('\tUnseen dataset size:', len(unseen_set))

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
forget_dataloader_train = torch.utils.data.DataLoader(forget_set_train, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
forget_dataloader_test = torch.utils.data.DataLoader(forget_set_test, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
retain_dataloader_train = torch.utils.data.DataLoader(retain_set_train, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
retain_dataloader_test = torch.utils.data.DataLoader(retain_set_test, batch_size=64, shuffle=False, num_workers=2,pin_memory=True)
unseen_dataloader = torch.utils.data.DataLoader(unseen_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

train_image_paths = TrainDataset().image_paths
test_image_paths = TestDataset().image_paths
forget_image_paths = ForgetDataset().image_paths
retain_image_paths = RetainDataset().image_paths
unseen_image_paths = UnseenDataset().image_paths

# Test the trained original model to serve as the base model for performing Machine Unlearning
unlearned_model = load_unlearned_model(weights_path)
print('\nEvaluating Base Model to unlearn')
test_acc = evaluation(unlearned_model, test_dataloader)
unseen_acc = evaluation(unlearned_model, unseen_dataloader)
mia = cal_mia(unlearned_model.cuda())
print(f'Test Acc: {test_acc}')
print(f'Unseen Acc: {unseen_acc}')
print(f'MIA: {mia}\n')

# Test the model trained only in retain data

retrained_model = models.resnet18(pretrained=False)
num_features = retrained_model.fc.in_features
retrained_model.fc = MultiLabelHead(num_features, 3)
retrained_model = retrained_model.cuda()
model_path_retrained = weights_path + 'retrained_model_weights.pth'
retrained_model.load_state_dict(torch.load(model_path_retrained))

print('Evaluating Trained only in retain without forget')
test_acc = evaluation(retrained_model, test_dataloader)
unseen_acc = evaluation(retrained_model, unseen_dataloader)
mia = cal_mia(retrained_model.cuda())
kl = compute_kl_divergence(retrained_model, unlearned_model, forget_dataloader_test)

print(f'Test Acc: {test_acc}')
print(f'Unseen Acc: {unseen_acc}')
print(f'MIA: {mia}')
print(f'KL-Div w.r.t Retrained model: {kl}\n')

#finetuning()
#cf_k()
#negative_grad()
#advanced_neg_grad()
#unsir()
#scrub()
ewcu()
#bad_teacher()