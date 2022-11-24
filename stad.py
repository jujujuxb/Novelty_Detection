import torch
from model import R_Net,D_Net,build_encoder
import argparse
from TrajectoriesDataSet import TrajectoryDataset
import os
from rich import print
from rich.progress import track
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_score, recall_score, roc_auc_score, log_loss, auc


def pred_labels(losses,threhold):

    pred = losses

    pred[np.where(losses <= threhold)] = 0
    pred[np.where(losses > threhold)] = 1
    
    return pred


def metrics_calalute(labels, losses):

    fpr, tpr, _ = roc_curve(labels, losses)

    img_auc = auc(fpr, tpr)

    fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]

    fig = plt.figure()

    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = {:.2f})'.format(
        img_auc), lw=3)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.legend()
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    # fpr95的指标
    fpr95 = tpr[np.where(fpr >= 0.95)[0][0]]

    png_root_path = os.path.join(os.getcwd(), 'RecurImg', "ocgan")

    if not os.path.exists(png_root_path):
        os.mkdir(png_root_path)

    # plt.(os.path.join(png_root_path, "{}_res.png".format("ocgan")))
    plt.show()

    figure = plt.figure()

    idx_0 = 0

    idx_1 = 0

    for loss, label in zip(losses, labels):
        if label == 0:
            plt.scatter(idx_0, loss, c='b', linewidths=0.1)
            idx_0 = idx_0 + 1
        else:
            plt.scatter(idx_1, loss, c='r', linewidths=0.1)
            idx_1 = idx_1 + 1

    # plt.savefig('loss.png')
    plt.show()
    
    pred = pred_labels(losses,0.03)

    p = precision_score(labels, pred,pos_label=0)

    r = recall_score(labels, pred,pos_label=0)

    f1 = f1_score(labels, pred,pos_label=0)

    print("fpr95:{},img_auc:{:.2f},precision:{:.2f},recall:{:.2f},f1:{:.2f}".format(
        fpr95, img_auc, p, r, f1))



def main(args):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    train_dataset = TrajectoryDataset(
        dataset_dir=args.data_path, labels={1, 2, 3, 4, 5, 6, 15})

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size,num_workers=1)

     
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    r_net_path = os.path.join(args.load_path, args.r_load_path)

    assert not r_net_path is None

    r_net = torch.load(r_net_path).to(device)
    print(f'Loaded R_Net from {r_net_path}')
    
    encoder_path = os.path.join(os.getcwd(),"models/encoder.pth")

    if os.path.exists(encoder_path):
        encoder = torch.load(encoder_path)
        print("Eocoder init from {}".format(encoder_path))
    else:
        encoder = build_encoder(3,64)

    
    r_net.cuda()
    encoder.cuda()
    
    critation = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    
    r_net.eval()
    for epoch in track(range(args.epochs)):
        total_iter = 0
        total_loss = 0
        for data,y in train_loader:
            x = data.cuda()
            optimizer.zero_grad()
            tea_pred = r_net(x)[1]
            stu_pred = encoder(x)
            loss = critation(tea_pred,stu_pred)
            loss.backward()
            optimizer.step()
            
            total_iter = total_iter + 1
            total_loss = total_loss +loss.item()
            
        print("Epoch:{},loss:{}".format(epoch,(total_loss/total_iter)))
    
        torch.save(encoder,"models/encoder.pth")

def test(args):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    in_dataset = TrajectoryDataset(
        dataset_dir=args.test_path, labels={1, 2, 3, 4, 5, 6, 15})

    in_dataorder = torch.utils.data.DataLoader(
        in_dataset, shuffle=False, batch_size=1)

     
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    r_net_path = os.path.join(args.load_path, args.r_load_path)

    assert not r_net_path is None

    r_net = torch.load(r_net_path).to(device)
    print(f'Loaded R_Net from {r_net_path}')
    
    encoder_path = os.path.join(os.getcwd(),"models/encoder.pth")

    assert os.path.exists(encoder_path)
    encoder = torch.load(encoder_path)
    print("Eocoder init from {}".format(encoder_path))
    
    r_net.cuda()
    encoder.cuda()
    
    critation = torch.nn.MSELoss()
    
    losses = []
    labels = []
    
    r_net.eval()
    encoder.eval()
    for data,y in in_dataorder:
        x = data.cuda()
        tea_pred = r_net(x)[1]
        stu_pred = encoder(x)
        loss = critation(tea_pred,stu_pred)
        losses.append(loss.item())
        labels.append(0)

    out_dataset = TrajectoryDataset(
        dataset_dir=args.data_path, labels={7, 8, 9, 10, 11, 12, 13, 14})

    out_dataloader = torch.utils.data.DataLoader(
        out_dataset, shuffle=False, batch_size=1)

    for data,y in out_dataloader:
        x = data.cuda()
        tea_pred = r_net(x)[1]
        stu_pred = encoder(x)
        loss = critation(tea_pred,stu_pred)
        losses.append(loss.item())
        labels.append(1)


    metrics_calalute(np.array(labels),np.array(losses))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', action='store_false',
                        help='Turns off gpu training')
    parser.add_argument('--save_path', '-sp', type=str, default='.',
                        help='Path to a folder where metrics and models will be saved')
    parser.add_argument('--d_save_path', '-dsp', type=str,
                        default='d_net.pth', help='Name of .pth file for d_net to be saved')
    parser.add_argument('--r_save_path', '-rsp', type=str,
                        default='r_net.pth', help='Name of .pth file for r_net to be saved')
    parser.add_argument('--load_path', '-lp', default='models',
                        help='Path to a folder from which models will be loaded')
    parser.add_argument('--d_load_path', '-dlp', type=str, default='d_net.pth',
                        help='Name of .pth file for d_net to be loaded')
    parser.add_argument('--r_load_path', '-rlp', type=str, default='r_net.pth',
                        help='Name of .pth file for r_net to be loaded')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for data loader')
    parser.add_argument('--rec_bound', type=float, default=0.1,
                        help='Upper bound of reconstruction loss')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--nw', type=int, default=1,
                        help='num_workers for DataLoader')
    parser.add_argument('--sstep', type=int, default=5,
                        help='Step in epochs for saving models')
    parser.add_argument('--std', type=float, default=0.155,
                        help='Standart deviation for noise in R_Net')
    parser.add_argument('--lambd', type=float, default=0.8,
                        help='Lambda parameter for LR loss')
    parser.add_argument('--cat', action='store_true',
                        help='Turns on skip connections with concatenation')
    parser.add_argument('--res', action='store_true',
                        help='Turns on residual connections')

    parser.add_argument('--data_path', type=str, default="/home/juxiaobing/code/GraduactionCode/data/T15/train",
                        help='Train Dataset\'s path')

    parser.add_argument('--test_path', type=str, default="/home/juxiaobing/code/GraduactionCode/data/T15/test",
                        help='Test Dataset\'s path')

    parser.add_argument('--odatapath', type=str, default="/home/juxiaobing/code/GraduactionCode/data/T15/T15_images",
                        help='Test Dataset\'s path')
    
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Test Dataset\'s path')

    args = parser.parse_args()

    test(args)


