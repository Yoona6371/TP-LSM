import time
import argparse
from torch.autograd import Variable
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from utils import *
from apmeter import APMeter
import os


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-gpu', type=str, default='1')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epoch', type=str, default='50')
parser.add_argument('-model', type=str, default='')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-batch_size', type=str, default='False')
parser.add_argument('-num_clips', type=str, default='False')
parser.add_argument('-skip', type=str, default='False')
parser.add_argument('-num_layer', type=str, default='False')
parser.add_argument('-unisize', type=str, default='False')
parser.add_argument('-alpha_l', type=float, default='1.0')
parser.add_argument('-beta_l', type=float, default='1.0')
args = parser.parse_args()

# set random seed
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED:', SEED)


batch_size = int(args.batch_size)
torch.backends.cudnn.enabled = False

if args.dataset == 'charades':
    # from charades_dataloader import Charades as Dataset

    from muti_dataloader import Charades as Dataset

    if str(args.unisize) == "True":
        print("uni-size padd all T to",args.num_clips)
        # from charades_dataloader import collate_fn_unisize

        from muti_dataloader import collate_fn_unisize

        collate_fn_f = collate_fn_unisize(args.num_clips)
        collate_fn = collate_fn_f.charades_collate_fn_unisize
    else:
        # from charades_dataloader import mt_collate_fn as collate_fn
        
        from muti_dataloader import mt_collate_fn as collate_fn


    # train_split = './data/charades.json'
    train_split = './data/multithumos.json'
    # train_split = './data/TSU_CS.json'
    test_split = train_split
    # rgb_root = '/home/data/Charades24fps/feature3'
    rgb_root =  '/home/data/MultiTHUMOS/feature' 
    # rgb_root =  '/home/data/TSU/RGB_i3d_16frames_64000_SSD' 
    flow_root = '/flow_feat_path/' # optional
    # rgb_of=[rgb_root, flow_root]
    # classes = 157
    # classes = 51
    classes = 65

def load_data(train_split, val_split, root):
    # Load Data
    print('load data', root)

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, classes, int(args.num_clips))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, classes, int(args.num_clips))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root
    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    
    return dataloaders , datasets


def run(models, num_epochs=50):
    since = time.time()
    Best_val_map = 0.
    for epoch in range(num_epochs):
        since1 = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            _, _ = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], epoch)
            sched.step(val_loss)
            # Time
            print("epoch", epoch, "Total_Time",time.time()-since, "Epoch_time",time.time()-since1)
            
            if Best_val_map < val_map:
                Best_val_map = val_map
                print("epoch",epoch,"Best Val Map Update",Best_val_map)
                pickle.dump(prob_val, open('./experiment/save/multithumos2/' + str(epoch) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                print("logit_saved at:","./experiment/save/multithumos2/" + str(epoch) + ".pkl")


def run_network(model, data, gpu, epoch=0, baseline=False):
    inputs, mask, labels, other = data
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)

    outputs_final, loss_f = model(inputs, labels)

    probs_f = torch.sigmoid(outputs_final) * mask.unsqueeze(2)

    loss = torch.sum(loss_f) / torch.sum(mask)

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data

        loss.backward()
        optimizer.step()

    train_map = 100 * apm.value().mean()
    print('epoch',epoch,'train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter

    return train_map, epoch_loss


def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    sampled_apm= APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        if sum(data[1].numpy()[0])>25:
            p1,l1=sampled_25(probs.data.cpu().numpy()[0],data[2].numpy()[0],data[1].numpy()[0])
            sampled_apm.add(p1,l1)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data
        
        probs_1 = mask_probs(probs.data.cpu().numpy()[0],data[1].numpy()[0]).squeeze()

        full_probs[other[0]] = probs_1.T

    epoch_loss = tot_loss / num_iter
    # torch.nonzero(100 * apm.value()).size()[0]  k 个类别里有几个类别动作在 n 组 视频中存在
    # apm.value() 1xK 张量，每个类 k 的平均精度
    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    sample_val_map = torch.sum(100 * sampled_apm.value()) / torch.nonzero(100 * sampled_apm.value()).size()[0]

    print('epoch',epoch,'Full-val-map:', val_map)
    print('epoch',epoch,'sampled-val-map:', sample_val_map)
    print(100 * sampled_apm.value())
    apm.reset()
    sampled_apm.reset()
    return full_probs, epoch_loss, val_map


if __name__ == '__main__':
    if args.mode == 'flow':
        print('flow mode', flow_root)
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'rgb':
        print('RGB mode', rgb_root)
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)

    if not os.path.exists('./experiment/save/multithumos2'):
        os.makedirs('./experiment/save/multithumos2')

    if args.train:

        if args.model == "TP_LSM":
            print("\TP_LSM")
            from nets.TPLSM import TPLSM
            num_clips = int(args.num_clips)
            # C
            num_classes = classes
            # D = 256, gamma = 1.5 
            inter_channels=[864, 576, 384, 256]

            s_size = [128, 128, 64, 64]
            # B
            num_block = 3
            # theta
            mlp_ratio = 6
            # D_0
            in_feat_dim = 1024
            # D_v
            final_embedding_dim = 512
            
            rgb_model = TPLSM(inter_channels, s_size, num_block, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes)

            print("loaded", args.load_model)

        rgb_model.cuda(1)

        lr = float(args.lr)
        optimizer = optim.Adam(rgb_model.parameters(), lr=lr)
        # patience=7
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=25, verbose=True)
        run([(rgb_model, 1, dataloaders, optimizer, lr_sched, args.comp_info)], num_epochs=int(args.epoch))
