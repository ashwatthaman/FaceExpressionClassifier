from rnnpre import RNNLM,Args
from chainer import Chain,optimizers,serializers
from chainer import links as L
from chainer import functions as F
from util.NNCommon import *
import util.generators as gens
from util.vocabulary import Vocabulary
import os,pandas
import numpy as np

class RNNClassifier(Chain):

    def __init__(self,args,class_n,drop_ratio=0.5):
        super(RNNClassifier, self).__init__(
            rnn_f = RNNLM(args.n_vocab,args.layer,args.embed,args.hidden,False),
            rnn_b = RNNLM(args.n_vocab,args.layer,args.embed,args.hidden,True),
            h2y = L.Linear(2*args.hidden,class_n)
        )
        self.n_vocab = args.n_vocab
        self.n_embed = args.embed
        self.n_layers = args.layer

    def __call__(self,xs,tag):
        h_f = self.rnn_f.getFeature(xs)
        h_b = self.rnn_b.getFeature(xs)
        h = F.concat((h_f,h_b),axis=1)
        y = self.h2y(h.data)
        loss = F.softmax_cross_entropy(y,tag)
        return loss

    def predict(self,xs):
        h_f = self.rnn_f.getFeature(xs)
        h_b = self.rnn_b.getFeature(xs)
        h = F.concat((h_f,h_b),axis=1)
        y = self.h2y(h.data)
        tag_arr = [y_each.argmax() for y_each in y.data]
        return tag_arr


    def loadModel(self,model_name_f,model_name_b):
        serializers.load_npz(model_name_f, self.rnn_f)
        serializers.load_npz(model_name_b, self.rnn_b)

    def setEpochNow(self,epoch_now):
        self.epoch_now = epoch_now

    def setMaxEpoch(self,epoch):
        self.epoch = epoch

    def setBatchSize(self,batch_size):
        self.batch_size = batch_size
        self.rnn_f.setBatchSize(batch_size)
        self.rnn_b.setBatchSize(batch_size)

    def setVocab(self,vocab):
        self.vocab = vocab

def train(args,tag_file):

    vocab_file = "vocab.bin"
    categ_file = "categ.bin"
    src_vocab = Vocabulary.load(vocab_file)

    if os.path.exists(categ_file):
        categ_dict = Vocabulary.load(categ_file)
        categ_len = len(categ_dict)
    else:
        categ_set = set()
        categ_len = len([categ_set.add("".join(char_arr)) for char_arr in gens.word_list(tag_file)])
        categ_dict = Vocabulary.new(gens.word_list(tag_file),categ_len)
        categ_dict.save(categ_file)

    rnn_classi = RNNClassifier(args,categ_len)
    optimizer = optimizers.Adam()
    optimizer.setup(rnn_classi)
    rnn_classi.setBatchSize(args.batchsize)
    rnn_classi.setVocab(src_vocab)
    rnn_classi.setMaxEpoch(args.epoch)
    rnn_classi.loadModel("./model/rnn_model_4_10.npz","./model/rnn_back4_10.npz")
    for e_i in range(args.epoch):
        total_loss = 0
        tt_now = ([src_vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list(args.source))
        tt_gen = gens.batch(tt_now,args.batchsize)

        # categ_now = ([categ_dict.stoi(char) for char in char_arr] for char_arr in gens.word_list(tag_file))
        categ_now = (categ_dict.stoi("".join(char_arr)) for char_arr in gens.word_list(tag_file))
        categ_gen = gens.batch(categ_now,args.batchsize)
        for tt,categ in zip(tt_gen,categ_gen):
            categ = np.array(categ,dtype=np.int32)
            loss = rnn_classi(tt,categ)
            total_loss+=loss.data

            rnn_classi.cleargrads()
            loss.backward()
            optimizer.update()

        print("saving... rnn_model_{}".format(e_i))
        print("total_loss:{}".format(total_loss))
        serializers.save_npz('../model/rnn_classi{}.npz'.format(e_i), rnn_classi)
        # serializers.save_npz(model_name.format(e_i,ti//save_iter), encdec)

def test(args,tag_file):

    vocab_file = "vocab.bin"
    categ_file = "categ.bin"
    src_vocab = Vocabulary.load(vocab_file)

    categ_dict = Vocabulary.load(categ_file)
    categ_len = len(categ_dict)

    rnn_classi = RNNClassifier(args,categ_len)
    optimizer = optimizers.Adam()
    optimizer.setup(rnn_classi)
    rnn_classi.setBatchSize(args.batchsize)
    rnn_classi.setVocab(src_vocab)
    rnn_classi.setMaxEpoch(args.epoch)
    serializers.load_npz("../model/rnn_classi39.npz",rnn_classi)
    tt_now = ([src_vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list(args.source))
    tt_gen = gens.batch(tt_now,args.batchsize)

    categ_now = (categ_dict.stoi("".join(char_arr)) for char_arr in gens.word_list(tag_file))
    categ_gen = gens.batch(categ_now,args.batchsize)
    tupl_arr = [('normal', "普通"),('komaru_c', "困る_口開け"),('yorokobi', "微笑"),('ikari_c', "怒り_口閉じ"),('gimon', "疑問"),('warai', "笑顔"),('komaru_o', "困る_口閉じ"),('odoroki', "驚き_大"),('ikari_o', "怒り_口開け"),('akire', "呆れ"),('nigawarai', "苦笑い"),('ero', "喘ぎ"),('doya', "ドヤ顔"),('odoroki_c', "驚き_小"),('itai', "苦痛"),('kangae', "考え中"),('haji', "恥じらい"),('naki', "泣き"),('awate', "慌てる"),('else', "それ以外"),('yorokobi_k', "恍惚"),('batu', "目バッテン"),('shock', "ショック"),('jito', "ジト目"),('ando', "安堵"),('uxtu', "うっ"),('chu', "キス"),('kowa', "恐れ"),('neboke', "寝ぼけ"),('normal_o', "普通_口開け"),('taibou', ""),('comi_ikari', ""),('warai_batu', "")]
    tag_hash = {tupl[0]:tupl[1] for tupl in tupl_arr}

    categ_list = [];tag_list=[]
    for tt,categ in zip(tt_gen,categ_gen):
        tag_arr = rnn_classi.predict(tt)
        categ_list+=[tag_hash[categ_dict.itos(categ_e)] for categ_e in categ]
        tag_list+=[tag_hash[categ_dict.itos(tag)] for tag in tag_arr]
        for tt_each,tag,categ_e in zip(tt,tag_arr,categ):
            print("{}, {}".format(tag_hash[categ_dict.itos(tag)],"".join([src_vocab.itos(tt_i) for tt_i in tt_each])))
    countFScore(categ_list,tag_list)

def loadModel(args,model_name="../model/rnn_classi39.npz"):

    vocab_file = "vocab.bin"
    categ_file = "categ.bin"
    src_vocab = Vocabulary.load(vocab_file)

    categ_dict = Vocabulary.load(categ_file)
    categ_len = len(categ_dict)

    rnn_classi = RNNClassifier(args,categ_len)
    optimizer = optimizers.Adam()
    optimizer.setup(rnn_classi)
    rnn_classi.setBatchSize(args.batchsize)
    rnn_classi.setVocab(src_vocab)
    rnn_classi.setMaxEpoch(args.epoch)
    serializers.load_npz(model_name,rnn_classi)
    return rnn_classi,src_vocab,categ_dict

def predict(args):
    rnn_classi,src_vocab,categ_dict = loadModel(args)

    tt_now = ([src_vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list(args.source))
    tt_gen = gens.batch(tt_now,args.batchsize)

    tupl_arr = [('normal', "普通"),('komaru_c', "困る_口開け"),('yorokobi', "微笑"),('ikari_c', "眉怒り_口閉じ"),('gimon', "疑問"),('warai', "笑顔"),('komaru_o', "困る_口閉じ"),('odoroki', "驚き_大"),('ikari_o', "眉怒り_口開け"),('akire', "呆れ"),('nigawarai', "苦笑い"),('ero', "喘ぎ"),('doya', "ドヤ顔"),('odoroki_c', "驚き_小"),('itai', "苦痛"),('kangae', "考え中"),('haji', "恥じらい"),('naki', "泣き"),('awate', "慌てる"),('else', "それ以外"),('yorokobi_k', "恍惚"),('batu', "目バッテン"),('shock', "ショック"),('jito', "ジト目"),('ando', "安堵"),('uxtu', "うっ"),('chu', "キス"),('kowa', "恐れ"),('neboke', "寝ぼけ"),('normal_o', "普通_口開け"),('taibou', ""),('comi_ikari', ""),('warai_batu', "")]
    tag_hash = {tupl[0]:tupl[1] for tupl in tupl_arr}

    for tt in tt_gen:
        tag_arr = rnn_classi.predict(tt)
        for tt_each,tag in zip(tt,tag_arr):
            print("{}, {}".format(tag_hash[categ_dict.itos(tag)],"".join([src_vocab.itos(tt_i) for tt_i in tt_each])))


def countFScore(teach_arr,pred_arr):
    fsc_hash = {chara:[0,0,0,0] for chara in set(teach_arr+pred_arr)}
    for teach,pred in zip(teach_arr,pred_arr):
        if teach==pred:
            for chara in fsc_hash:
                if chara==teach:fsc_hash[chara][0]+=1
                else:fsc_hash[chara][3]+=1
        else:
            fsc_hash[teach][1]+=1
            fsc_hash[pred][2]+=1
    prec_arr = []
    reca_arr = []
    fscr_arr = []
    for chara in fsc_hash:
        if fsc_hash[chara][0]==0:continue
        prec = round(fsc_hash[chara][0]/(fsc_hash[chara][0]+fsc_hash[chara][1]),3)
        reca = round(fsc_hash[chara][0]/(fsc_hash[chara][0]+fsc_hash[chara][2]),3)
        f_score = round(2*prec*reca/(prec+reca),3)
        prec_arr.append(prec);reca_arr.append(reca)
        fscr_arr.append(f_score)
        # print("|{}|{}|{}|".format(chara,f_score,fsc_hash[chara]))
        print("|{}|{}|".format(chara,f_score))
    print("|minor_f|{}|".format(round(sum(fscr_arr)/len(fscr_arr),3)))
    major_f = 2*sum(prec_arr)*sum(reca_arr)/(len(prec_arr)*(sum(prec_arr)+sum(reca_arr)))
    print("|major_f|{}|".format(round(major_f,3)))
    print("|精度|{}|".format(len([1 for teach,pred in zip(teach_arr,pred_arr) if teach==pred])/len(pred_arr)))


if __name__=="__main__":
    args = Args()
    # args.source = "../EmotionClassifier/data/serif_spm_train.txt"
    # args.epoch=40
    # train(args,tag_file="../EmotionClassifier/data/tag_train.txt")
    args.source = "../EmotionClassifier/data/serif_spm_test.txt"
    test(args,tag_file="../EmotionClassifier/data/tag_test.txt")