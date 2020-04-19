#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch
from pytorch_transformers import BertTokenizer

from models import  model_builder
from models.model_builder import Bert, ExtSummarizer

from others.logging import logger, init_logger

from torchsummary import summary
from torch.autograd import Variable


model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


import torch
import torch.nn as nn
import numpy as np

from models.encoder import PositionalEncoding, ExtTransformerEncoder
from models.decoder import TransformerDecoderLayer

from pytorch2keras import pytorch_to_keras


MAX_SIZE = 5000





def load_my_state_dict(new_model, state_dict):

    own_state = new_model.state_dict()
    for name, param in state_dict.items():
        if name.replace('bert.', '') not in own_state:
              continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name.replace('bert.', '')].copy_(param)


def load_my_state_dict_decoder(new_model, state_dict):
    ## replace ext_layer by decoder to make it work with AbsSummarizer decoder
    own_state = new_model.state_dict()
    for name, param in state_dict.items():
        if name.replace('ext_layer.', '') not in own_state:
              continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
            print('okkkk')
        own_state[name.replace('ext_layer.', '')].copy_(param)

def only_model(args, device_id):

    logger.info('Loading checkpoint from %s' % args.test_from)
    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    

    ### We load our ExtSummarizer model
    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    ### We create an encoder and a decoder like those of ExtSummarizer and load the latter parameters into the former
    ### This is for the test sake
    encoder  = Bert(False, '/tmp', True)
    load_my_state_dict(encoder, checkpoint['model'])

    decoder = ExtTransformerEncoder(encoder.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
    load_my_state_dict_decoder(decoder, checkpoint['model'])

    encoder.eval()
    decoder.eval()

    seq_len = 250


    ### We test if the parameters have been well loaded
    input_ids = torch.tensor([np.random.randint(100, 15000, seq_len)], dtype=torch.long)
    mask = torch.ones(1, seq_len, dtype=torch.float)
    clss = torch.tensor([[20,36,55, 100, 122, 130, 200, 222]], dtype=torch.long)
    mask_cls = torch.tensor([[1]*len(clss[0])], dtype=torch.long)

    """## test encoder
    top_vec = model.bert(input_ids, mask)
    top_vec1 = encoder(input_ids, mask)
    logger.info((top_vec-top_vec1).sum())

    ## test decoder
    sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
    sents_vec = sents_vec * mask_cls[:, :, None].float()

    sents_vec1 = top_vec1[torch.arange(top_vec1.size(0)).unsqueeze(1), clss]
    sents_vec1 = sents_vec1 * mask_cls[:, :, None].float()


    scores = model.ext_layer(sents_vec, mask_cls)
    scores1  = decoder(sents_vec1, mask_cls)
    logger.info((scores-scores1).sum())"""




    ################# ONNX ########################"
    
    ## Now we are exporting the encoder and the decoder into onnx 
    """input_names = ["input_ids", "mask"]
    output_names = ["hidden_outputs"]
    torch.onnx.export(model.bert.to('cpu'), (input_ids, mask), "/tmp/encoder5.onnx", verbose=True, 
                      input_names=input_names, output_names=output_names, export_params=True, keep_initializers_as_inputs=True)"""

    k_model = pytorch_to_keras(model.bert.to('cpu'), [input_ids, mask], [(1,250,), (1,250,)], verbose=True)  

    print("okkk")


    """logger.info("Load onnx and test")
    model_enc = onnx.load("/tmp/encoder.onnx")
    # Check that the IR is well formed
    logger.info(onnx.checker.check_model(model_enc))

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model_enc.graph))

    tf_rep = prepare(onnx_model)  # prepare tf representation
    print(tf_rep)
    tf_rep.export_graph("/tmp/m.pb")  # export the model"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=4, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)



    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger()
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    only_model(args, device_id)