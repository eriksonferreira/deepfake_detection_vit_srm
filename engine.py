# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train and eval functions used in main.py

Mostly copy-paste from https://github.com/facebookresearch/deit/blob/main/engine.py
"""

import math
from typing import Iterable, Optional
import torch

from timm.data import Mixup
from timm.utils import accuracy
from einops import rearrange

import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.BCEWithLogitsLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=True,
                    finetune=False
                    ):
    
    
    criterion = torch.nn.BCEWithLogitsLoss()
    if finetune:
        model.train(not finetune)
    else:
        model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1060

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        batch_size = targets.size(0)

        samples = samples.to(device, non_blocking=True)
        targets = targets.unsqueeze(1)
        targets = targets.float()
        targets = targets.to(device, non_blocking=True)

        

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(samples)
            # Verificar e remover NaNs dos outputs e labels
            mask = ~torch.isnan(outputs).view(-1)
            outputs = outputs[mask]
            targets = targets[mask]
            
            # Verificar se todos os outputs são válidos
            if outputs.numel() == 0:
                print("Todos os outputs são NaNs, pulando este batch.")
                continue

            loss = criterion(outputs, targets)
            loss_value = loss.item()
            # print(loss_value)

            if not math.isfinite(loss_value):
                print("OUTPUT", outputs)
                print("TARGET", targets)
                print("Loss is {}, stopping training".format(loss_value))
                raise ValueError("Loss is {}, stopping training".format(loss_value))

            optimizer.zero_grad()

            # print(f"Outputs type: {outputs.dtype}, Targets type: {targets.dtype}")

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            if amp:
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
            else:
                loss.backward(create_graph=is_second_order)
                if max_norm is not None and max_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, world_size, distributed=True, amp=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.unsqueeze(1)
        target = target.float()
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast(enabled=amp):
            output = model(images)
            # print("output", output)

        if distributed:
            outputs.append(concat_all_gather(output))
            targets.append(concat_all_gather(target))
        else:
            outputs.append(output)
            targets.append(target)

    num_data = len(data_loader.dataset)
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    real_acc1, real_acc5 = accuracy(outputs[:num_data], targets[:num_data], topk=(1, 1))
    acc = accuracy_new(outputs[:num_data], targets[:num_data])
    print("NEW ACCURACY: ", acc)
    real_loss = criterion(outputs, targets)
    # print(f"Loss: {real_loss.item()}") 
    # precision_val, _ = precision(outputs[:num_data], targets[:num_data])
    recall_val = recall(outputs[:num_data], targets[:num_data])
   
    metric_logger.update(loss=real_loss.item())
    metric_logger.meters['acc1'].update(real_acc1.item())
    metric_logger.meters['acc5'].update(real_acc5.item())
    metric_logger.meters['new_acc'].update(acc)
    metric_logger.meters['new_acc_raw'].update(acc/100)
    metric_logger.meters['recall'].update(recall_val.item())
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(f"Recall: {recall_val}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    if tensor.dim() == 1:
        output = rearrange(tensors_gather, 'n b -> (b n)')
    else:
        output = rearrange(tensors_gather, 'n b c -> (b n) c')
    return output


# def precision(output, target):
#     """Computes the precision for binary classification"""
#     with torch.no_grad():
#         pred = torch.round(torch.sigmoid(output)).to(torch.int64)  # Converte logits em 0 ou 1 e depois para inteiros
#         target = target.to(torch.int64)  # Converte target para inteiros
#         true_positives = (pred & target).sum().float()  # AND bitwise para contar verdadeiros positivos
#         predicted_positives = pred.sum().float()  
#         precision_score = true_positives / (predicted_positives + 1e-8)  # Evita divisão por zero
#     return precision_score


def recall(output, target):
    """Computes the recall for binary classification"""
    # print(output)
    # print("#############3")
    # print(target)
    with torch.no_grad():
        pred = torch.round(torch.sigmoid(output)).float()  # Converte logits em 0 ou 1     
        # print("##########################")
        # print(pred)
        pred = pred.bool()  # Converte para booleano
        target = target.bool()  # Converte para booleano

        true_positives = (pred & target).sum().float()  # AND bitwise para contar verdadeiros positivos
        actual_positives = target.sum().float()  # Todos os positivos reais

        recall_score = true_positives / (actual_positives + 1e-8)  # Evita divisão por zero
    return recall_score


def accuracy_new(output, target):
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        pred = torch.round(torch.sigmoid(output)).long()  # Converte logits em 0 ou 1
        correct = pred.eq(target).float()  # Compara com os rótulos verdadeiros
        accuracy_score = correct.sum() * 100.0 / target.size(0)  # Calcula a acurácia
    return accuracy_score.item()
