import torch
import torchvision
import numpy as np
import matplotlib.pylab as plt
import os
from sklearn.metrics import roc_curve, auc



def plot_auc(output,label,epoch,loss,fig_path,n_classes = 3,plot_show=False):
    fpr = dict()
    tpr = dict()
    roc_auc = []
    # y_test = np.squeeze(np.array(label))
    # y_score = np.squeeze(np.array(output))
    y_test = np.array(label).reshape(-1,n_classes)
    y_score = np.array(output).reshape(-1, n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc.append(auc(fpr[i], tpr[i]))
    mean_auc = np.mean(roc_auc[:4])

    # Plot of a ROC curve for a specific class
    if plot_show:
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='Class %d ROC curve (area = %0.2f)' % (i+1,roc_auc[i]))
            plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('%d epoch Receiver operating characteristic example \r\n (loss = %0.3f,mean_auc = %0.3f)'%(epoch,loss,mean_auc))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(fig_path,'epoch_%dth.jpg'%epoch))
        plt.show()
    return roc_auc

def calculate_accuracy(outputs, targets):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    hit = ((outputs > 0.5) == targets).sum()
    #hit = sum(abs(outputs-targets))
    tsum = targets.shape[0]
    return (hit + 1e-8) / (tsum + 1e-8)

def add_image(inputs, outputs, targets, writer, subset, epoch,name):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    for h in range(targets.shape[0]):
        data = []
        data.append(inputs[h, :, :, :])
        data = [x for x in data]
        data = torch.cat(data, dim=1)
        # data_all = torchvision.utils.make_grid(data, nrow=1, padding=2, normalize=False, range=None, scale_each=False)
        data_all = torchvision.utils.make_grid(data, nrow=int(np.ceil(np.sqrt(len(data)))), padding=10, normalize=True,
                                               range=None, scale_each=True)
        writer.add_image(subset + '_step_' + str(epoch) +'/'+ name + '/gt: ' + str(targets[h]) + '/pred: ' + str(outputs[h]),
                         img_tensor=data_all, global_step=epoch, dataformats='CHW')


def add_image_3d(inputs, outputs, targets, writer, subset, epoch,name):

    outputs = outputs.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    c_sl = 12
    # print('image added... with len of {}'.format(len(targets)))
    data = []
    for h in range(targets.shape[0]):
        data.append(inputs[h, 0:3, c_sl, :, :])
    data = [x for x in data]
    #data = torch.cat(data, dim=0)
    data_all = torchvision.utils.make_grid(data, nrow=int(np.ceil(np.sqrt(len(data)))), padding=10, normalize=True, range=None, scale_each=True)
    if subset == 'val':
        writer.add_image(subset + '_step_' + str(epoch) +'/'+ name + '/Diff_'+str(sum(sum(abs(outputs-targets)))) + '/diff_'+str(sum(abs(outputs-targets))) + '/gt:' + str(targets) + '/pred:' + str(outputs),
                     img_tensor=data_all, global_step=epoch, dataformats='CHW')
    else:
        writer.add_image(
            subset + '_step_' + str(epoch) +'/'+ name + '/Diff_' + str(sum(sum(abs(outputs - targets)))) + '/diff_' + str(
                sum(abs(outputs - targets))),img_tensor=data_all, global_step=epoch, dataformats='CHW')



def add_gl_image(images,patches, outputs, targets, writer, subset, epoch):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    for h in range(targets.shape[0]):
        data_g = []
        data_l = []
        data_g.append(images[h, :, :, :])
        data_l.append(patches[h, :, :, :])
        data_g = [x for x in data_g]
        data_l = [x for x in data_l]
        data_g = torch.cat(data_g, dim=1)
        data_l = torch.cat(data_l, dim=1)
        data_g_all = torchvision.utils.make_grid(data_g, nrow=1, padding=2, normalize=False, range=None, scale_each=False)
        data_l_all = torchvision.utils.make_grid(data_l, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        writer.add_image(subset + '_step_' + str(epoch) + '/diff_'+str(abs(outputs[h]-targets[h])) + '_g_/gt: ' + str(targets[h]) + '/pred: ' + str(outputs[h]),img_tensor=data_g_all, global_step=epoch, dataformats='CHW')
        writer.add_image(subset + '_step_' + str(epoch) + '/diff_'+str(abs(outputs[h]-targets[h])) + '_l_/gt: ' + str(targets[h]) + '/pred: ' + str(outputs[h]),img_tensor=data_l_all, global_step=epoch, dataformats='CHW')

def add_gld_image(images,patches,details, outputs, targets, writer, subset, epoch):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    for h in range(targets.shape[0]):
        data_g = []
        data_l = []
        data_d = []
        data_g.append(images[h, :, :, :])
        data_l.append(patches[h, :, :, :])
        data_d.append(details[h, :, :, :])
        data_g = [x for x in data_g]
        data_l = [x for x in data_l]
        data_d = [x for x in data_d]
        data_g = torch.cat(data_g, dim=1)
        data_l = torch.cat(data_l, dim=1)
        data_d = torch.cat(data_d, dim=1)
        data_g_all = torchvision.utils.make_grid(data_g, nrow=1, padding=2, normalize=False, range=None, scale_each=False)
        data_l_all = torchvision.utils.make_grid(data_l, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        data_d_all = torchvision.utils.make_grid(data_d, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        writer.add_image(subset + '_step_' + str(epoch) + '/diff_'+str(abs(outputs[h]-targets[h])) + '_g_/gt: ' + str(targets[h]) + '/pred: ' + str(outputs[h]),img_tensor=data_g_all, global_step=epoch, dataformats='CHW')
        writer.add_image(subset + '_step_' + str(epoch) + '/diff_'+str(abs(outputs[h]-targets[h])) + '_l_/gt: ' + str(targets[h]) + '/pred: ' + str(outputs[h]),img_tensor=data_l_all, global_step=epoch, dataformats='CHW')
        writer.add_image(
            subset + '_step_' + str(epoch) + '/diff_' + str(abs(outputs[h] - targets[h])) + '_d_/gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_d_all, global_step=epoch, dataformats='CHW')

def add_gl_image_index(images, patches, outputs, targets, writer, subset, epoch,index):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    for h in range(targets.shape[0]):
        data_g = []
        data_l = []
        data_g.append(images[h, :, :, :])
        data_l.append(patches[h, :, :, :])
        data_g = [x for x in data_g]
        data_l = [x for x in data_l]
        data_g = torch.cat(data_g, dim=1)
        data_l = torch.cat(data_l, dim=1)
        data_g_all = torchvision.utils.make_grid(data_g, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        data_l_all = torchvision.utils.make_grid(data_l, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        writer.add_image(
            subset + '_step_' + str(epoch)+ '_diff_' + str(outputs[h] - targets[h]) + '_index_' + str(index) + '/g_gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_g_all, global_step=epoch,
            dataformats='CHW')
        writer.add_image(
            subset + '_step_' + str(epoch)+ '_diff_' + str(outputs[h] - targets[h]) + '_index_' + str(index) + '/l_gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_l_all, global_step=epoch,
            dataformats='CHW')

def add_gld_image_index(images, patches, details, outputs, targets, writer, subset, epoch,index):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    for h in range(targets.shape[0]):
        data_g = []
        data_l = []
        data_d = []
        data_g.append(images[h, :, :, :])
        data_l.append(patches[h, :, :, :])
        data_d.append(details[h, :, :, :])
        data_g = [x for x in data_g]
        data_l = [x for x in data_l]
        data_d = [x for x in data_d]
        data_g = torch.cat(data_g, dim=1)
        data_l = torch.cat(data_l, dim=1)
        data_d = torch.cat(data_d, dim=1)
        data_g_all = torchvision.utils.make_grid(data_g, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        data_l_all = torchvision.utils.make_grid(data_l, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        data_d_all = torchvision.utils.make_grid(data_d, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        writer.add_image(
            subset + '_step_' + str(epoch)+ '_diff_' + str(outputs[h] - targets[h]) + '_index_' + str(index) + '/g_gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_g_all, global_step=epoch,
            dataformats='CHW')
        writer.add_image(
            subset + '_step_' + str(epoch)+ '_diff_' + str(outputs[h] - targets[h]) + '_index_' + str(index) + '/l_gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_l_all, global_step=epoch,
            dataformats='CHW')
        writer.add_image(
            subset + '_step_' + str(epoch) + '_diff_' + str(outputs[h] - targets[h]) + '_index_' + str(
                index) + '/d_gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_d_all, global_step=epoch,
            dataformats='CHW')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

