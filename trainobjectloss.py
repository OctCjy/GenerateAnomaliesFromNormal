import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR100
from torchvision.datasets import ImageFolder
from model.utils import Reconstruction3DDataLoader, Reconstruction3DDataLoaderJump
from model.autoencoder import *
from utils import *
from Yolov3.mynewdetect import*

import argparse


parser = argparse.ArgumentParser(description="AutoEncoder")
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--epochs', type=int, default=130, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate phase 1')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2-mask','ped2','avenue', 'shanghai'], help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='basename of folder to save weights')

parser.add_argument('--model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')

parser.add_argument('--pseudo_anomaly_mask', type=float, default=0.2, help='mask')
parser.add_argument('--object_loss_weight', type=float, default=0.5, help='object_loss_weight')
parser.add_argument('--print_all', action='store_true', help='print all reconstruction loss')



args = parser.parse_args()

# assert 1 not in args.jump

exp_dir = args.exp_dir
exp_dir += 'lr' + str(args.lr) if args.lr != 1e-4 else ''
exp_dir += 'weight'
exp_dir += '_recon'
exp_dir += '_pamask' + str(args.pseudo_anomaly_mask) if args.pseudo_anomaly_mask != 0 else ''
exp_dir += '_objloss_' + str(args.object_loss_weight) if args.object_loss_weight != 0 else ''

print('exp_dir: ', exp_dir)

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

train_folder = os.path.join(args.dataset_path, args.dataset_type, 'training', 'frames')

# Loading dataset
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'

if  args.pseudo_anomaly_mask>0 :
    # cifar_transform = transforms.Compose([
    #             transforms.RandomCrop(32, padding=12, padding_mode='reflect'),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomVerticalFlip(),
    #             transforms.ToTensor()
    # ])
    cifar_dataset = CIFAR100('dataset/cifar100', transform=transforms.ToTensor())
    cifar_batch = data.DataLoader(cifar_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)
    cifar_iter = iter(cifar_batch)

train_dataset = Reconstruction3DDataLoader(train_folder, transforms.Compose([transforms.ToTensor()]),
                                                resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, return_normal_seq=True, img_extension=img_extension)

train_size = len(train_dataset)

train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, drop_last=True)
# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'a')
sys.stdout = f

torch.set_printoptions(profile="full")

loss_func_mse = nn.MSELoss(reduction='none')

##############################################################################
yolo = Darknet(cfg='Yolov3/yolov3/cfg/yolov3-spp.cfg')
weights = 'Yolov3/yolov3/weights/yolov3-spp-ultralytics.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo.load_state_dict(torch.load(weights, map_location=device)['model'])
yolo.to(device).eval()
#############################################################################



if args.start_epoch < args.epochs:
    model = convAE()
    model = nn.DataParallel(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # resume
    if args.model_dir is not None:
        #assert args.start_epoch > 0
        # Loading the trained model
        model_dict = torch.load(args.model_dir)
        model_weight = model_dict['model']
        model.load_state_dict(model_weight.state_dict())
        optimizer.load_state_dict(model_dict['optimizer'])
        model.cuda()

    # model.eval()
    for epoch in range(args.start_epoch, args.epochs):
        pseudolossepoch = 0
        lossepoch = 0
        pseudolosscounter = 0
        losscounter = 0

        for j, (imgs) in enumerate(train_batch):

            net_in = copy.deepcopy(imgs)
            net_in = net_in.cuda()

            cifar_inpainting_smoothborder_pseudo_stat = []
            mask_inpainting_mask_pseudo_stat = []
            cls_labels = []

            for b in range(args.batch_size):
                rand_number = np.random.rand()

                #yolo pseudo anomaly but with inpainting loss
                pseudo_anomaly_mask = (0 <= rand_number <  args.pseudo_anomaly_mask)
                #total_pseudo_prob += args.pseudo_anomaly_jump_inpainting
                if pseudo_anomaly_mask:
                    try:
                        # Samples the batch
                        cifar_img, _ = next(cifar_iter)
                    except StopIteration:
                        # restart the generator if the previous generator is exhausted.
                        cifar_iter = iter(cifar_batch)
                        cifar_img, _ = next(cifar_iter)

                    with torch.no_grad():
                        #cifar_img[0].shape=torch.Size([3, 32, 32]),0-1
                        dataset_tenosr=LoadTensor(copy.deepcopy(net_in[b]))
                        #maskedvideo=detect(yolo,dataset_tenosr,save_img=False,cifar_img=None)
                        maskedvideo,bboxes=detect(yolo,dataset_tenosr,save_img=False,cifar_img=cifar_img[0])
                        mask_inpainting_mask_pseudo_stat.append(bboxes)
                        

                    net_in[b] = maskedvideo.cuda()#yolov3+mask
                    
                    '''
                    for i in range (16):
                        saveimgs = (net_in[b,:,i].cpu().detach().numpy() + 1) * 127.5
                        saveimgs = saveimgs.transpose(1,2,0).astype(dtype=np.uint8)
                        cv2.imwrite(os.path.join(args.vid_dir,'IN_{:04d}_{:04d}_{:04d}_mask.png').format(j,b,i), saveimgs)
                    
                    for i in range (16):
                        saveimgs = (imgs[b,:,i].cpu().detach().numpy() + 1) * 127.5
                        saveimgs = saveimgs.transpose(1,2,0).astype(dtype=np.uint8)
                        cv2.imwrite(os.path.join(args.vid_dir,'IN_{:04d}_{:04d}_{:04d}.png').format(j,b,i), saveimgs)
                    '''
                else:
                    mask_inpainting_mask_pseudo_stat.append(None)
                    
                rand_number = np.random.rand()
       

            ########## TRAIN GENERATOR
            outputs = model.forward(net_in)

            cls_labels = torch.Tensor(cls_labels).unsqueeze(1).cuda()

            loss_mse = loss_func_mse(outputs, net_in)

            object_level_loss=0

            modified_loss_mse = []

            for b in range(args.batch_size):
                if mask_inpainting_mask_pseudo_stat[b] is not None:                      
                    ###############################################################################
                    modified_loss_mse.append(torch.mean(loss_func_mse(outputs[b], imgs[b].to(outputs.device))))
                    object_level_loss += get_bbox_loss(imgs[b].to(outputs.device),outputs[b],mask_inpainting_mask_pseudo_stat[b])             
                    pseudolossepoch += modified_loss_mse[-1].cpu().detach().item()
                    pseudolosscounter += 1

                    '''              
                    for i in range (16):
                        saveimgs = (net_in[b,:,i].cpu().detach().numpy() + 1) * 127.5
                        saveimgs = saveimgs.transpose(1,2,0).astype(dtype=np.uint8)
                        cv2.imwrite(os.path.join(args.vid_dir,'{:04d}_{:04d}_{:04d}_in.png').format(j,b,i), saveimgs)
                    
                    for i in range (16):
                        saveimgs = (outputs[b,:,i].cpu().detach().numpy() + 1) * 127.5
                        saveimgs = saveimgs.transpose(1,2,0).astype(dtype=np.uint8)
                        cv2.imwrite(os.path.join(args.vid_dir,'{:04d}_{:04d}_{:04d}_out.png').format(j,b,i), saveimgs)
                    '''

                else:
                    modified_loss_mse.append(torch.mean(loss_mse[b]))
                    lossepoch += modified_loss_mse[-1].cpu().detach().item()
                    losscounter += 1  
                

            assert len(modified_loss_mse) == loss_mse.size(0)
            stacked_loss_mse = torch.stack(modified_loss_mse)
            
            loss = torch.mean(stacked_loss_mse)+args.object_loss_weight*object_level_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if j % 5 == 0 or args.print_all:
                print("epoch {:d} iter {:d}/{:d}".format(epoch, j, len(train_batch)))
                print('Loss: {:.6f}'.format(loss.item()))

        print('----------------------------------------')
        print('Epoch:', epoch)
        if pseudolosscounter != 0:
            print('PseudoMeanLoss: Reconstruction {:.9f}'.format(pseudolossepoch/pseudolosscounter))
        if losscounter != 0:
            print('MeanLoss: Reconstruction {:.9f}'.format(lossepoch/losscounter))

        # Save the model and the memory items
        model_dict = {
            'model': model,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(model_dict, os.path.join(log_dir, 'model_{:03d}.pth'.format(epoch)))

print('Training is finished')
sys.stdout = orig_stdout
f.close()



