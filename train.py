import sys
from optparse import OptionParser
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from model import ResNet, eval_net_loader, make_checkpoint_dir, normlize_slice
from lib.utils import fft_downsize
from lib import plot_net_predictions
import matplotlib
# matplotlib.use('QT5Agg')
from matplotlib import pyplot as plt
plt.ion()

from dataloader import *
from piqa import SSIM


class SSIMLoss(SSIM):
    def __init__(self, **kwargs):
        super().__init__(n_channels=1, **kwargs)

    def forward(self, x, y):
        return 1 - super().forward(x, y)


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def forward(self, output, target):
        crit = nn.MSELoss()
        return crit(output, target)*0.01


def train_epoch(epoch, train_dataset_detail, criterion, optimizer, batch_size, n_channel, normslice, scheduler):
    
    net.train()
    epoch_loss = 0

    lr_imgs = torch.empty(batch_size, n_channel, 160, 160)#LR image 80,80
    hr_imgs = torch.empty(batch_size, 1, 160, 160) #HR image 160,160

    file_names_train = train_dataset_detail['file_names_train']
    random.shuffle(file_names_train)

    train_path = train_dataset_detail['train_path']
    train_num = int(train_dataset_detail['train_num'])//10 #only use half for training for each epoch

    total_iter = len([i for i in range(0, train_num, batch_size)])
    iter_num = 0
    for index in range(0, train_num-batch_size+1, batch_size):  # train_num
        iter_num = iter_num+1
        for i in range(0, batch_size):
            with np.load(os.path.join(train_path, file_names_train[index+i])) as sample:
                lr_imgs[i, :, :, :] = torch.from_numpy(sample['lr'])
                hr_imgs[i, :, :] = torch.from_numpy(sample['hr'])

        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        outputs = net(lr_imgs)
        hr_imgs = hr_imgs.clone().detach()

        loss = criterion[0](outputs, hr_imgs)
        loss2 = criterion[1](outputs, lr_imgs[:, 0:1])
        epoch_loss += loss.item()
        epoch_loss += loss2.item()

        # print(f'epoch = {epoch+1:d}, iteration = {iter_num:d}, loss = {(loss.item()+loss2.item())/batch_size:.5f}')

        # save to summary
        if iter_num %40==0:
            writer.add_scalar('train_loss_iter', 
                                  loss.item()+loss2.item(),
                                  iter_num + total_iter * epoch)
            # writer.add_figure('predictions vs. actuals',
            #                       plot_net_predictions(lr_imgs, hr_imgs, masks_pred, batch_size),
            #                       global_step = iter_num + total_iter * epoch)

        optimizer.zero_grad()
        (loss+loss2).backward()
        optimizer.step()
    print(f'Epoch finished ! Loss: {epoch_loss/(batch_size*total_iter):.5f}, lr:{scheduler.get_last_lr()}')
    # loss_plot.append(epoch_loss / train_num)



def validate_epoch(epoch,batch_size,train_dataset_detail,n_classes,n_channel,normslice,device,criterion):

    precision, fidelity = eval_net_loader(net, train_dataset_detail, n_classes, batch_size, n_channel,normslice,criterion,device)
    return precision, fidelity
 

def train_net(train_dataset_detail,  net, device, model_name, normslice, CE, n_channel =1, epochs=5, batch_size=1, lr=0.1, save_cp=5):

    print(f'''
    Starting training:
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {lr}
        Training size: {train_dataset_detail['length_dataset']}
        Validation size: {train_dataset_detail['length_dataset']-train_dataset_detail['train_num']}
        Checkpoints: {str(save_cp)}
        Device: {str(device)}
    ''')
    loss_plot = []
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)
    criterion = SSIMLoss().to(device)#nn.L1Loss()#   SSIMLoss().to(device)#nn.L1Loss()#nn.MSELoss()
    criterion = []
    criterion.append(nn.L1Loss())
    criterion.append(ConsistencyLoss())

    best_precision = np.Inf
    best_fidelity = np.Inf
    for epoch in range(epochs):
          
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        train_epoch(epoch, train_dataset_detail, criterion, optimizer, batch_size, n_channel, normslice, scheduler)
        scheduler.step()

        if save_cp and epoch % save_cp == 0:
            acc_loss, fidelity = validate_epoch(epoch, batch_size, train_dataset_detail, 1, n_channel, normslice, device, criterion)
            loss_plot.append(acc_loss)
            plt.plot(loss_plot)
            plt.show()
            # if precision < best_precision and fidelity < 100:
            state_dict = net.state_dict()
            if device == "cuda":
                state_dict = net.state_dict()
            torch.save(state_dict, dir_checkpoint+model_name+f'-lr{lr}'+f'Ba{batch_size}.pth')
            print('Checkpoint {} saved !'.format(epoch + 1))
            # best_precision = precision
            # best_fidelity = fidelity
        if len(loss_plot)>7 and (loss_plot[-1]>loss_plot[-7]):
            break
    writer.close()

    
def get_args():
          
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-b', dest='batchsize', default=8, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.05,
                      type='float', help='learning rate')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-f', '--folder', dest='folder', 
                      default='', help='folder name')
    parser.add_option('--model_name', type=str, default='Res2UNET_SR_L1001L2_TEST')
    parser.add_option('--n_channel',  default=4, type='int', help='batch size')

    parser.add_option('--celoss', default=False, help='cross entropy loss or not')
    parser.add_option('--pre_model_folder', type=str, default='..')#NoNormNoClip
    parser.add_option('--pre_model_name', type=str, default='SR_L1-lr0.1Ba16')
    parser.add_option('--normslice', default=False, help='normalize slice')


###
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    print('torch.cuda.is_available()')
    print(torch.cuda.is_available())
    print('device')
    print(device)
    args = get_args()
    print(args)
    
    dir_data = f'./data/{args.folder}'
    dir_checkpoint = f'./checkpoints/{args.folder}/'#_b{args.batchsize}
    dir_summary = f'./runs/{args.folder}_b{args.batchsize}'
    params = {'batch_size': args.batchsize, 'shuffle': True, 'num_workers': 4}

    print(dir_checkpoint)
    make_checkpoint_dir(dir_checkpoint)
    writer = SummaryWriter(dir_summary)
    
    val_ratio = 0.1
    train_dataset_detail = dataset_detail(val_ratio)

    net = ResNet(n_channels=args.n_channel, reps=2)
    net.to(device)

    if args.load:
        model_path = './checkpoints/' + args.pre_model_name + '.pth'  # WeitCE-lr0.2Ba32epo66.pth'
        net.load_state_dict(torch.load(model_path))
        print('Model loaded from {}'.format(model_path))
    net.to(device)
    
    # train model in parallel on multiple-GPUs
    if torch.cuda.device_count() > 1:
        print("Model training on", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net) 

    try:
        train_net(train_dataset_detail, net, device, args.model_name, normslice=args.normslice, CE=args.celoss,
                  n_channel=args.n_channel, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
