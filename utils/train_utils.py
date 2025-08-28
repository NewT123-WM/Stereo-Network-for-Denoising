from .SNR_RSNR import ComputeSNR
import torch
import math

def train_one_epoch(model, device, trainloader, optimizer, criterion):
    model.train()
    train_loss, train_snr = 0.0, 0.0
    
    for inputs, labels in trainloader:
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_snr += ComputeSNR(outputs.squeeze(), labels.squeeze())

    return train_loss / len(trainloader.dataset), train_snr / len(trainloader.dataset)





def validate(model, device, testloader, criterion):
    model.eval()
    test_loss, test_snr = 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in testloader:        
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.unsqueeze(1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            test_snr += ComputeSNR(outputs.squeeze(), labels.squeeze())

    return test_loss / len(testloader.dataset), test_snr / len(testloader.dataset)