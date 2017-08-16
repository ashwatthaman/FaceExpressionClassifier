import subprocess
from rnnclassifier import loadModel
from rnnpre import Args

class EmotClassifier():

    def __init__(self,args):
        rnn_classi, src_vocab, categ_dict = loadModel(args)
        self.rnn_classi = rnn_classi
        self.src_vocab = src_vocab
        self.categ_dict = categ_dict

        tupl_arr = [('normal', "普通"), ('komaru_c', "困る_口開け"), ('yorokobi', "微笑"), ('ikari_c', "眉怒り_口閉じ"),
                    ('gimon', "疑問"), ('warai', "笑顔"), ('komaru_o', "困る_口閉じ"), ('odoroki', "驚き_大"),
                    ('ikari_o', "眉怒り_口開け"), ('akire', "呆れ"), ('nigawarai', "苦笑い"), ('ero', "喘ぎ"), ('doya', "ドヤ顔"),
                    ('odoroki_c', "驚き_小"), ('itai', "苦痛"), ('kangae', "考え中"), ('haji', "恥じらい"), ('naki', "泣き"),
                    ('awate', "慌てる"), ('else', "それ以外"), ('yorokobi_k', "恍惚"), ('batu', "目バッテン"), ('shock', "ショック"),
                    ('jito', "ジト目"), ('ando', "安堵"), ('uxtu', "うっ"), ('chu', "キス"), ('kowa', "恐れ"), ('neboke', "寝ぼけ"),
                    ('normal_o', "普通_口開け"), ('taibou', ""), ('comi_ikari', ""), ('warai_batu', "")]
        self.tag_hash = {tupl[0]: tupl[1] for tupl in tupl_arr}

    def __call__(self,text_arr):
        word_arr = [self.separateToMorph(text) for text in text_arr]
        out_arr = self.classifyEmot(word_arr)
        return out_arr

    def separateToMorph(self,text):
        text_morph = subprocess.check_output("echo \"{}\"　| spm_encode --model=serif.model".format(text), shell=True)
        text = text_morph.decode("utf-8")
        word_arr = text.replace("▁","").strip().split(" ")
        return word_arr

    def classifyEmot(self,text_arr):
        tt = [[self.src_vocab.stoi(word) for word in word_arr] for word_arr in text_arr]
        tag_arr = self.rnn_classi.predict(tt)
        out_arr = []
        for tt_each, tag in zip(tt, tag_arr):
            # print("{}, {}".format(self.tag_hash[self.categ_dict.itos(tag)], "".join([self.src_vocab.itos(tt_i) for tt_i in tt_each])))
            out_arr.append("{}, {}".format(self.tag_hash[self.categ_dict.itos(tag)], "".join([self.src_vocab.itos(tt_i) for tt_i in tt_each])))
            # out_arr.append(self.tag_hash[self.categ_dict.itos(tag)])
        return out_arr


if __name__=="__main__":
    args = Args()
    ec = EmotClassifier(args)
    text_arr = ["あんたなんかにわからないだろうけど！"]
    text_arr.append("えへへ。変なことになっちゃったね。")
    text_arr.append("はぁ……。何言ってるの？")
    out_arr = ec(text_arr)
    print("表情,セリフ")
    for out in out_arr:
        print(out)

