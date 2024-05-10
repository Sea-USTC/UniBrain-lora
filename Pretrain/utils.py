import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return  self.total
        else:
            return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
            )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # args.local_rank = os.environ['LOCAL_RANK']
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

# def init_distributed_mode(args):

#     # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#     #     args.rank = int(os.environ["RANK"])
#     #     args.world_size = int(os.environ['WORLD_SIZE'])
#     #     args.gpu = int(os.environ['LOCAL_RANK'])
#     # elif 'SLURM_PROCID' in os.environ:
#     #     args.rank = int(os.environ['SLURM_PROCID'])
#     #     args.gpu = args.rank % torch.cuda.device_count()
#     # else:
#     #     print('Not using distributed mode')
#     #     args.distributed = False
#     #     return
#     # rank = int(os.environ['RANK'])                # system env process ranks\
#     # print(torch.distributed.get_world_size())
    
#     args.distributed = True
#     # torch.cuda.set_device(args.gpu)
#     num_gpus = torch.cuda.device_count()          # Returns the number of GPUs available
#     torch.cuda.set_device(args.rank % num_gpus)
#     # args.gpu = args.rank % torch.cuda.device_count()
    
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}'.format(
#         args.rank, args.dist_url), flush=True)
#     torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                          world_size=args.world_size, rank=args.rank)
#     torch.distributed.barrier()
#     print('using distributed mode',args.rank, args.dist_url)
#     setup_for_distributed(args.rank == 0)

# # export MASTER_ADDR=localhost
# # export MASTER_PORT=5678
    

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import openpyxl as op

def op_toexcel(result, filename): # openpyxl库储存数据到excel
    # result fid:{"pred":"","gt":""}
    wb = op.Workbook() # 创建工作簿对象
    ws = wb['Sheet'] # 创建子表
    ws.append(['fid','gt','pred']) # 添加表头
    for fid in result:
        d = fid, result[fid]["gt"], result[fid]["pred"]
        ws.append(d) # 每次写入一行
    wb.save(filename)

def check_pred(target_class,fids,threshs,pred,gt,strict_test,filename):
    result = {fid:{"pred":[],"gt":[]} for fid in fids}
    for i in range(len(target_class)):
        gt_np = gt[:, i].cpu().numpy()
        pred_np = pred[:, i].cpu().numpy()
        pred_label = (pred_np>=threshs[i])*1
        mask = (gt_np == -1).squeeze()
        if strict_test:
            gt_np[mask] = 0
            pred_label[mask] = 0
        else:
            gt_np[mask] = pred_label[mask]
        fid_pred = fids[pred_label==1]
        fid_real = fids[gt_np==1]

        for fid in fid_pred:
            result[fid]["pred"].append(target_class[i])
        for fid in fid_real:
            result[fid]["gt"].append(target_class[i])
    for fid in result:
        result[fid]["pred"].sort()
        result[fid]["gt"].sort()
        result[fid]["pred"] = ','.join(result[fid]["pred"])
        result[fid]["gt"] = ",".join(result[fid]["gt"])
    op_toexcel(result, filename)

def plot_auc_pr(GT_list,pred_list,dis,rootdir, mode):
    # pred_list and GT_list are assumed to be numpy arrays or lists
    rootdir = rootdir+"curve_"+mode+'/'
    os.makedirs(rootdir,exist_ok=True)
    fpr, tpr, _ = roc_curve(GT_list, pred_list)
    roc_auc = auc(fpr, tpr)
    index = 0
    for idx,i in enumerate(fpr):
        if i >= 0.1:
            index = idx
            break
    print(dis,"fp",fpr[index],tpr[index])

    precision, recall, _ = precision_recall_curve(GT_list, pred_list)
    pr_auc = auc(recall, precision)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve '+dis)
    plt.legend(loc="lower right")
    plt.savefig(rootdir+'roc_'+dis+'.png')
    # plt.show()

    # Plot PR curve
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve '+dis)
    plt.legend(loc="lower right")
    plt.savefig(rootdir+'pr_'+dis+'.png')
    # plt.show()