import argparse
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms




def main():

    parser = argparse.ArgumentParser(description='Toy example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda training')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='seeding number')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='checkpoints after as many iterations')
    parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')
    args = parser.parse_args()
    print(args)
    #use cuda when is available and enabled from the argument parameter
    use_cuda = torch.cuda.is_available() and not args.no_cuda

    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(args.seed)

    if use_cuda:
      # set the cuda seed for experiment repeatability
      torch.cuda.manual_seeds(args.seed)

    #Data 
    # pinned memory for easy access
    data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')
    train_data = datasets.MNIST(data_path, train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.361))]))
    test_data = datasets.MNIST(data_path, train=False, 
                               transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    # Model

    class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
          self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
          self.conv2_drop = nn.Dropout2d()
          self.fc1 = nn.Linear(320, 50)
          self.fc2 = nn.Linear(50,10)

      def forward(self, x):
          x = F.relu(F.max_pool2d(self.conv1(x),2))
          x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
          x = x.view(-1, 320)
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          return F.log_softmax(x, dim=1)

    # model init 
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.resume:
      model.load_state_dict(torch.load('model.pth'))
      optimizer.load_state_dict(torch.load('optimizer.pth'))
    # training

    model.train()
    train_losses = []

    for i, (data, target) in enumerate(train_loader):
        data = data.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

        if i%10 == 0:
         print(i, loss.item())
         torch.save(model.state_dict(), 'model.pth')
         torch.save(optimizer.state_dict(), 'optimizer.pth')
         torch.save(train_losses, 'train_losses.pth')


if __name__ == '__main__':
    main()
