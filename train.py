import numpy as np
import torch
from utils import data_pacing_function, generate_random_batch, get_num_correct, naive_pacing_function
import torch.optim as optim

def train_model(hyp_params, train_set, test_set, model):
    metrics = {"curriculum": "", "sample_size": 0, "train_accuracy":[] , 
        "test_accuracy":[], "train_loss":[], "test_loss":[], "batch":[]}
    metrics["curriculum"] = hyp_params.order
    ## Hyperparameters
    
    device = hyp_params.device
    batch_size = hyp_params.batch_size
    optimizer = optim.SGD(model.parameters(), lr=hyp_params.lr)
    criterion = hyp_params.criterion
    interval = hyp_params.interval
    batch_increase = hyp_params.batch_increase
    increment = hyp_params.increment
    starting_percentage = hyp_params.starting_percentage
    num_epochs = hyp_params.num_epochs

    current_pace = 1
    total_correct = 0
    running_num_samples = 0
    running_loss = 0.0
    test_img = [x for x,_ in test_set]
    test_labels = [y for _,y in test_set]
    #test_img, test_labels = test_img.to(device).float(), test_labels.to(device)
    X_train = [x for x,_ in train_set]
    y_train = [x for _,x in train_set]
    model  = model.to(device)
    
    for batch in range(num_epochs*len(X_train)):
        print("pace:",current_pace)  # loop over the dataset multiple times
        for i in range(len(train_set)):
            print(i)
            X, y, current_pace = data_pacing_function(train_set[i], batch_increase, increment, starting_percentage, batch, current_pace, batch_size)
            inputs, labels = generate_random_batch(X, y, X.shape[0])

            inputs, labels = inputs.to(device).float(), labels.to(device)
            labels = labels.squeeze()
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels.long())
            total_correct += get_num_correct(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_num_samples += len(labels)
            accuracy = total_correct/running_num_samples

        if batch % interval == interval-1:    # print every 2000 mini-batches
            print(f'[{batch + 1}] loss: {running_loss/running_num_samples:.3f} pace: {current_pace} data_size: {len(y)}')

        if batch % 400 == 399: 
            test_acc, test_loss = evaluate_model(test_img, test_labels, model, criterion)

            metrics["train_accuracy"].append(accuracy)
            metrics["train_loss"].append(running_loss/running_num_samples)
            metrics["test_accuracy"].append(test_acc/len(test_labels))
            metrics["test_loss"].append(test_loss)
            metrics["batch"].append(batch)

    return metrics


def evaluate_model(X_img, y_test, model, criterion):
    accuracy = 0
    loss = 0
    model.eval()
    X_img = torch.unsqueeze(X_img, 1)
    #y_test = torch.unsqueeze(y_test,1)
    with torch.no_grad():
        outputs = model(X_img)
        accuracy = get_num_correct(outputs, y_test)
        loss = criterion(outputs, y_test).item()
    model.train()
    return accuracy, loss
