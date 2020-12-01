import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import classifier

def main(args):
    input_dim = 28 * 28
    output_dim = 10

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    train_set = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    val_set = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False)

    linear_model = classifier.LinearClassifier(input_dim, output_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(linear_model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = Variable(inputs.view(-1, IN))
            labels = Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = linear_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            '''
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.
            '''

        else:
            '''
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i, running_loss / (i % 100)))
            '''

            linear_model.eval()

            with torch.no_grad():
                val_loss = 0.0
                val_corrects = 0
                for i, data in enumerate(val_loader, 0):
                    inputs, labels = data

                    inputs = Variable(inputs.view(-1, IN))
                    labels = Variable(labels)

                    outputs = linear_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    val_loss += criterion(outputs, labels).item()  # sum up batch loss
                    val_corrects += torch.sum(preds == labels.data)

                    # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    # correct += pred.eq(target.view_as(pred)).sum().item()

                epoch_loss = val_loss / (i + 1)
                epoch_acc = val_corrects.double() / len(val_set)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    'val', epoch_loss, epoch_acc))




    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--n_epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    main(args)

