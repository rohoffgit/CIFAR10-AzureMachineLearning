### use your own arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int)
parser.add_argument("--numepochs", type=int, default=2)
args, unknown_args = parser.parse_known_args()
debug = args.debug == 1
if debug:
    print('WARN: debug mode!')
print(f'Arguments: numepochs={args.numepochs}')


### get run from the current context
# https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py
from azureml.core import Run
run = Run.get_context()

### get data and train model
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import Net

# download CIFAR 10 data
# https://en.wikipedia.org/wiki/CIFAR-10
trainset = torchvision.datasets.CIFAR10(
    root='./data_train',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# download CIFAR 10 data
testset = torchvision.datasets.CIFAR10(
    root='./data_test',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=1,
    shuffle=True,
    num_workers=2
)


# define convolutional network
net = Net()
# set up pytorch loss /  optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# train the network
for epoch in range(args.numepochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # unpack the data
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            loss = running_loss / 2000
            # ADDITIONAL CODE: log loss metric to AML
            run.log('loss', loss)
            print(f'epoch={epoch + 1}, batch={i + 1:5}: loss {loss:.2f}')
            running_loss = 0.0

        # quick and dirty 
        if debug and i > 100: break

print('Finished Training')

### save model 
torch.save(net, 'outputs/model.pt')

### predict some samples
import os

# https://en.wikipedia.org/wiki/CIFAR-10
cls = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i, data in enumerate(testloader, 0):
    inputs, labels = data
    print(f'data: {inputs.size()}, {labels.size()}')
    
    with torch.no_grad():
        prediction = torch.max(net(inputs)[0], 0).indices
        label = labels[0]

    # export prediction
    pil_img = torchvision.transforms.ToPILImage()(inputs[0])
    folder = f'outputs/samples/{prediction == label}'
    if not os.path.exists(folder): os.makedirs(folder)
    pil_img.save(f'{folder}/{cls[label]}_{label}_{prediction}.png')
    if debug and i > 100: break