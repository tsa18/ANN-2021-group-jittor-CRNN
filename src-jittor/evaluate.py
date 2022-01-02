from argparse import ArgumentParser
from config import *
import os
import jittor as jt
from jittor import nn
from tqdm import tqdm

from dataset import Synth90kDataset,SVTDataset, IIITDataset
from model import CRNN
from ctc_decoder import ctc_decode
from config import evaluate_config as config


def evaluate(crnn, dataloader, criterion,
             max_iter=None, decode_method='beam_search', beam_size=10):
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with jt.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            images, targets, target_lengths = [d for d in data]

            logits = crnn(images)
            log_probs = nn.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = jt.int64([log_probs.size(0)] * batch_size)
            modified_targets=[]
            longest_len=target_lengths.max().item()
            s=0
            for i in target_lengths:
                i=i.item()
                target=targets[s:s+i]
                padding_len=longest_len-len(target)
                padding=jt.zeros(padding_len)
                target= jt.concat([target,padding],dim=0)
                modified_targets.append(target)
                s=s+i
            modified_targets=jt.stack(modified_targets,dim=0)
            modified_targets=modified_targets.int32()
            # print(modified_targets)
            targets = modified_targets
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs.numpy(), method=decode_method, beam_size=beam_size)
            reals = targets.numpy().tolist()
            target_lengths = target_lengths.numpy().tolist()
            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, real, target_length in zip(preds, reals, target_lengths):
                # real1 = reals[target_length_counter:target_length_counter + target_length]
                # print("real1",real1)
                real = real[:target_length]
                # target_length_counter += target_length
                # print("pred, real: ",pred," ,",real)
                if pred == real:
                    # print("here")
                    tot_correct += 1
                else:
                    # print("there")
                    wrong_cases.append((real, pred))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases
    }
    return evaluation


def main():
    eval_batch_size = config['eval_batch_size']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_height = config['img_height']
    img_width = config['img_width']

    jt.flags.use_cuda=jt.has_cuda

    #root_dir=config['data_dir']
    # root_dir='../data/IIIT/'
    root_dir='../data/SVT/'
    # root_dir = '../data/ICDAR2003/'
    # root_dir = '../data/ICDAR2013/'
    # test_dataset = IIITDataset(root_dir=root_dir, mode='test',
    #                                img_height=img_height, img_width=img_width,
    #                                batch_size=eval_batch_size,
    #                                 shuffle=False,
    #                                 num_workers=cpu_workers)
    test_dataset = SVTDataset(root_dir=root_dir, mode='test',
                                   img_height=img_height, img_width=img_width,
                                   batch_size=eval_batch_size,
                                    shuffle=False,
                                    num_workers=cpu_workers)

    num_class = len(SVTDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint[-3:] == ".pt":
        import torch
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location="cpu"))
    else:
        crnn.load(reload_checkpoint)

    criterion = jt.CTCLoss(reduction='sum')

    evaluation = evaluate(crnn, test_dataset, criterion, 10,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'])
    print('test_evaluation: loss={loss}, acc={acc}'.format(**evaluation))


if __name__ == '__main__':
    main()
