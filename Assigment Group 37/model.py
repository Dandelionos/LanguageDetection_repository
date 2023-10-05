import numpy as np
import torch
import torch.nn as nn
from IPython.display import Audio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


#Data augmentation, creates arrays of augmented data, saves them as .npy files
#Uses Audiomentations to perform augmentations
def augment():
    files = ['noiseaug.npy', 'tsaug.npy', 'psaug.npy', 'shiftaug.npy']

    X_train, y_train = np.load("dataset/inputs_train_fp16.npy"), np.load(
        "dataset/targets_train_int8.npy"
    )
    X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
        "dataset/targets_test_int8.npy"
    )

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    augmentations = [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5), 
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5), 
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5), 
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)]

    for augmentation, file in zip(augmentations, files):
        if file not in os.listdir():
            augment = OneOf([augmentation])
            augmented_X_train = augment(samples = X_train , sample_rate = 16000)
            np.save(file, augmented_X_train, allow_pickle = True)
        print(f'{file} done')

    some = SomeOf(2, augmentations)(samples = X_train, sample_rate = 16000)
    mix = Compose(augmentations)(samples = X_train, sample_rate = 16000)
    
    noiseTrain = np.load('noiseaug.npy')
    tsTrain = np.load('tsaug.npy')
    psTrain = np.load('psaug.npy')
    shiftTrain = np.load('shiftaug.npy')

    augmented_X_train = np.concatenate((X_train, noiseTrain, tsTrain, psTrain, shiftTrain, some, mix))
    augmented_y_train = np.concatenate((y_train, y_train, y_train, y_train, y_train, y_train, y_train))

    np.save('augmented_X_train.npy', augmented_X_train, allow_pickle=True)
    np.save('augmented_y_train.npy', augmented_y_train, allow_pickle=True)

#Full code to create an run model, with testing and training included
def run_model():
    X_train, y_train = np.load('augmented_X_train.npy'), np.load(
    	'augmented_y_train.npy'
    )

    X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
        "dataset/targets_test_int8.npy"
    )

    X_test = X_test.astype(np.float32)

    #Dataset class for loaders
    class MyDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, index):
            return self.X[index], self.y[index]

    training_data = MyDataset(X_train, y_train)
    test_data = MyDataset(X_test, y_test)

    #Custom collate function to improve batch processing and formatting of data
    def collate(batch):
    	tensors, targets = [], []

    	for data, label in batch:
    		data = torch.reshape(data, (1, 40000))
    		label = label.type(torch.LongTensor)
    		tensors += [data]
    		targets += [label]

    	tensors = torch.stack(tensors)
    	targets = torch.stack(targets)

    	return tensors, targets

    #Train and test loaders using the collate function
    batch_size = 256

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )

    #Model class based on nn.Module
    #4 Groups of convolution layers coupled with batch normalisation and pooling
    class CNN(nn.Module):
        def __init__(self, n_input=1, n_output=6, stride=16, n_channel=32):
            super().__init__()
            self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride, bias = False)
            self.bn1 = nn.BatchNorm1d(n_channel)
            self.pool1 = nn.MaxPool1d(4)
            self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3, bias = False)
            self.bn2 = nn.BatchNorm1d(n_channel)
            self.pool2 = nn.MaxPool1d(4)
            self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3, bias = False)
            self.bn3 = nn.BatchNorm1d(2 * n_channel)
            self.pool3 = nn.MaxPool1d(4)
            self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3, bias = False)
            self.bn4 = nn.BatchNorm1d(2 * n_channel)
            self.pool4 = nn.MaxPool1d(4)
            self.average_pool = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(2 * n_channel, n_output)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(self.bn1(x))
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(self.bn2(x))
            x = self.pool2(x)
            x = self.conv3(x)
            x = F.relu(self.bn3(x))
            x = self.pool3(x)
            x = self.conv4(x)
            x = F.relu(self.bn4(x))
            x = self.pool4(x)
            x = self.average_pool(x)
            x = x.permute(0, 2, 1)
            x = self.fc1(x)
            return F.log_softmax(x, dim=2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(n_input=1, n_output=6)
    model.to(device)

    learning_rate = 0.0002
    weight_decay = 0

    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    #Training loop using nll loss function and a tqdm progressbar for visualisation
    def train(model, epoch):
        model.train()
        for batch, (data, target) in enumerate(train_loader):

        	data = data.to(device)
        	target = target.to(device)
        	output = model(data)
        	loss = F.nll_loss(output.squeeze(), target)
        	
        	for param in model.parameters():
                param.grad = None

        	loss.backward()
        	optimizer.step()

        	pbar.update(pbar_update)
        	losses.append(loss.item())

    #Calculates number of correct predictions per batch
    def number_of_correct(pred, target):
    	return pred.squeeze().eq(target).sum().item()

    #Gets predictions for a batch
    def get_likely_index(tensor):
        return tensor.argmax(dim=-1)

    #Test loop using the functions above
    def test(model, epoch):
        model.eval()
        correct = 0
        for data, target in test_loader:

            data = data.to(device)
            target = target.to(device)
            
            output = model(data)

            pred = get_likely_index(output)

            correct += number_of_correct(pred, target)

            pbar.update(pbar_update)

        accuracy = 100. * correct / len(test_loader.dataset)

        print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n")

    n_epoch = 80

    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []

    with tqdm(total=n_epoch) as pbar:
        for epoch in range(1, n_epoch + 1):
            train(model, epoch)
            test(model, epoch)
            scheduler.step()

    model = torch.jit.script(model.to("cpu"))
    model.save(f'modelCE_lr{learning_rate}_wd{weight_decay}.pt')