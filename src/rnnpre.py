#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os
sys.path.append("../src/")
from chainer import Variable,optimizers,serializers,Chain
import chainer.links as L
import chainer.functions as F
import numpy as np
xp = np
from util.NNCommon import *
import util.generators as gens
from util.vocabulary import Vocabulary
import chainer

chainer.using_config('use_cudnn', 0)
chainer.config.use_cudnn =  "never"


class LSTM(L.NStepLSTM):

    def __init__(self,n_layer, in_size, out_size, dropout=0.5):
        n_layers = 1
        super(LSTM, self).__init__(n_layers, in_size, out_size, dropout)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(LSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM, self).to_gpu(device)
        if self.cx is not None:
            self.cx.to_gpu(device)
        if self.hx is not None:
            self.hx.to_gpu(device)

    def set_state(self, cx, hx):
        assert isinstance(cx, Variable)
        assert isinstance(hx, Variable)
        cx_ = cx
        hx_ = hx
        if self.xp == np:
            cx_.to_cpu()
            hx_.to_cpu()
        else:
            cx_.to_gpu()
            hx_.to_gpu()
        self.cx = cx_
        self.hx = hx_

    def reset_state(self):
        self.cx = self.hx = None

    def __call__(self, xs, train=True):
        batch = len(xs)
        if self.hx is None:
            xp = self.xp
            self.hx = Variable(
                    xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype))
        if self.cx is None:
            xp = self.xp
            self.cx = Variable(
                    xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype))

        hy, cy, ys = super(LSTM, self).__call__(self.hx, self.cx, xs)
        self.hx, self.cx = hy, cy
        return ys
    
    
class RNNLM(Chain):

    def __init__(self,vocab_size,layer_size,in_size,out_size,back_flag,drop_ratio=0.5):
        super(RNNLM, self).__init__(
            embed = L.EmbedID(vocab_size,in_size),
            dec = LSTM(layer_size,in_size, out_size, dropout=drop_ratio),
            h2w = L.Linear(out_size,vocab_size),
        )
        self.n_vocab = vocab_size
        self.n_embed = in_size
        self.n_layers = layer_size
        self.out_size = out_size
        self.back_flag = back_flag
    
    def loadW(self):
        src_vocab = self.vocab
        premodel_name=""
        src_w2ind = {}; trg_w2ind = {}
        src_ind2w = {}; trg_ind2w = {}
        src_size = self.n_vocab
        print(src_size)
        for vi in range(src_size):
            #src_ind2w[vi] = self.__src_vocab.itos(vi)
            # src_ind2w[vi] = src_vocab[vi]
            src_ind2w[vi] = src_vocab.itos(vi)
            src_w2ind[src_ind2w[vi]] = vi
        #self.xe.W.data = xp.array(transferWordVector(src_w2ind,src_ind2w,premodel_name),dtype=xp.float32)
        print("pre:{}".format(self.embed.W.data[0][:5]))
        self.embed.W = Variable(xp.array(transferWordVector(src_w2ind,src_ind2w,premodel_name),dtype=xp.float32))
        print("pos:{}".format(self.embed.W.data[0][:5]))
     
    def makeEmbedBatch(self,xs,reverse=False):
        if reverse:
            xs = [xp.asarray(x[::-1],dtype=xp.int32) for x in xs]
        elif not reverse:
            xs = [xp.asarray(x,dtype=xp.int32) for x in xs]
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre) # CuPy does not have cumsum()
        xs = F.split_axis(self.embed(F.concat(xs, axis=0)), sections, axis=0)
        return xs
    
    def setEpochNow(self,epoch_now):
        self.epoch_now = epoch_now

    def setMaxEpoch(self,epoch):
        self.epoch = epoch

    def setBatchSize(self,batch_size):
        self.batch_size = batch_size

    def setVocab(self,vocab):
        self.vocab = vocab
    
    def reset_state(self):
        self.dec.reset_state()
        
    def __call__(self,xs):
        self.reset_state()
        #xs = [x for x in xs]#1は<s>を指す。decには<s>から入れる。</s>まで予測する。
        if not self.back_flag:
            t_pred = [x[1:]+[2] for x in xs]#1は<s>を指す。decには<s>から入れる。
            x_vec = self.makeEmbedBatch(xs)
        else:
            t_pred = [x[::-1][1:]+[1] for x in xs]
            x_vec = self.makeEmbedBatch(xs,True)

        ys_d = self.dec(x_vec)
        ys_w = self.h2w(F.concat(ys_d, axis=0))
        t_all = []
        for t_each in t_pred: t_all += t_each
        t_all = xp.array(t_all, dtype=xp.int32)
        loss = F.softmax_cross_entropy(ys_w, t_all)  # /len(t_all)
        if len(t_pred[0])%10==0:
            print("t:{}".format([self.vocab.itos(tp_e) for tp_e in t_pred[0]]))
            print("y:{}\n".format([self.vocab.itos(int(ys_w.data[ri].argmax())) for ri in range(len(t_pred[0]))]))
        # [print(y_w.data.shape) for y_w in ys_w]
        return loss

    def getFeature(self,xs):
        self.reset_state()
        #xs = [x for x in xs]#1は<s>を指す。decには<s>から入れる。</s>まで予測する。
        if not self.back_flag:
            t_pred = [x[1:]+[2] for x in xs]#1は<s>を指す。decには<s>から入れる。
            x_vec = self.makeEmbedBatch(xs)
        else:
            t_pred = [x[::-1][1:]+[1] for x in xs]
            x_vec = self.makeEmbedBatch(xs,True)
        ys_d = self.dec(x_vec)
        batch_size = len(ys_d)
        ys_d = F.concat([y_d[-1] for y_d in ys_d],axis=0)
        ys_d = F.reshape(ys_d,(batch_size,self.out_size))
        return ys_d


    def predict(self,xs,vocab):
        t = [1]*len(xs)#1は<s>を指す。decには<s>から入れる。</s>まで予測する。
        xs = [x for x in xs]#1は<s>を指す。decには<s>から入れる。

        t = [self.embed(xp.array([t_each],dtype=xp.int32)) for t_each in t]
        xs_f = self.makeEmbedBatch(xs)
        xs_b = self.makeEmbedBatch(xs,True)

        self.enc_f.reset_state()
        self.enc_b.reset_state()
        ys_f = self.enc_f(xs_f,train=False)
        ys_b = self.enc_b(xs_b,train=False)

        self.dec.hx = self.enc_f.hx
        self.dec.cx = self.enc_f.cx
        ys_d = self.dec(t,train=False)
        ys_w = [self.h2w(y) for y in ys_d]
        # print("ys_w:{}".format(len(ys_w)))
        # print("ys_w:{}".format(ys_w[0].data.shape))
        print([ys.data for ys in ys_w])
        t = [(y_each.data[-1].argmax(0)) for y_each in ys_w]
        print("t:{}".format([vocab.itos(t_each) for t_each in t]))
        t = [self.embed(xp.array([t_each],dtype=xp.int32)) for t_each in t]
        # pred_arr = [src_vocab.itos(t_each) for t_each in t]
        count_len=0
        while count_len<10:
            ys_d = self.dec(t,train=False)
            ys_w = [self.h2w(y) for y in ys_d]
            t = [(y_each.data[-1].argmax(0)) for y_each in ys_w]
            print("t:{}".format([vocab.itos(t_each) for t_each in t]))
            t = [self.embed(xp.array([t_each],dtype=xp.int32)) for t_each in t]
            count_len+=1

    def loadModel(self,model_name_base,args):
        first_e = 0
        model_name = ""
        for e in range(args.epoch):
            model_name_tmp = model_name_base.format(args.dataname, args.dataname, e,args.n_latent)
            if os.path.exists(model_name_tmp):
                model_name = model_name_tmp
                self.setEpochNow(e + 1)

        if os.path.exists(model_name):
            print(model_name)
            serializers.load_npz(model_name, self)
            print("loaded_{}".format(model_name))
            first_e = self.epoch_now
        else:
            print("loadW2V")
            if os.path.exists(args.premodel):
                self.loadW(args.premodel)
            else:
                print("wordvec model doesnt exists.")
        return first_e
    

class Args():
    def __init__(self):
        self.source =""
        self.epoch = 5
        self.n_vocab = 16000#19079
        self.embed = 300
        self.back_flag = True
        self.hidden= 1200
        self.layer = 1
        self.batchsize=30
        self.dropout = 0.5
        self.gpu = -1
        if self.gpu>-1:
            global xp
            import cupy as xp
        self.gradclip = 5

def train(args):
    encdec = RNNLM(args.n_vocab,args.layer,args.embed,args.hidden,args.back_flag)
    optimizer = optimizers.Adam()
    optimizer.setup(encdec)
    vocab_file = "vocab.bin"
    if os.path.exists(vocab_file):
        src_vocab = Vocabulary.load(vocab_file)
    else:
        src_vocab = Vocabulary.new(gens.word_list(args.source), args.n_vocab)
        src_vocab.save(vocab_file)
    print("loadW2V")
    encdec.setBatchSize(args.batchsize)
    encdec.setVocab(src_vocab)
    encdec.setMaxEpoch(args.epoch)
    if args.back_flag:
        model_name = "rnn_back{}_{}.npz"
    else:
        model_name = "rnn_model{}_{}.npz"
                 
    #encdec.loadW()
    if args.gpu>-1:
        encdec.to_gpu()
    
    save_iter = 10000
    for e_i in range(args.epoch):
        total_loss = 0
        loss = None
        tt_now = ([src_vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list(args.source))
        tt_gen = gens.batch(tt_now,args.batchsize)
        for ti,tt in enumerate(tt_gen):
            try:
                loss = encdec(tt)
                total_loss+=loss.data
            except IndexError:
                print("IndexError:{}".format(ti))
                print("   {}".format(tt))
                continue
            except chainer.utils.type_check.InvalidType:
                print("InvalidType:{}".format(ti))
                print("   {}".format(tt))
                continue

            
            encdec.cleargrads()
            loss.backward()
            optimizer.update()

            if ti%save_iter==save_iter-1:                
                print("saving... rnn_model_{}_{}".format(e_i,ti//save_iter))
                print("total_loss:{}".format(total_loss))
                serializers.save_npz(model_name.format(e_i,ti//save_iter), encdec)


if __name__=="__main__":
    args = Args()
    train(args)



