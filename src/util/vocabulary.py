from collections import defaultdict
import codecs

class Vocabulary:
  def __init__(self):
    pass

  def __len__(self):
    return self.__size

  def stoi(self, s):
    #if s not in self.__stoi:print("notin:{}".format(s))
    return self.__stoi[s]

  def itos(self, i):
    return self.__itos[i]

  @staticmethod
  def new(list_generator, size):
    self = Vocabulary()
    self.__size = size

    word_freq = defaultdict(lambda: 0)
    for words in list_generator:
      for word in words:
        word_freq[word] += 1

    # for word in word_freq:
    #     print("{}:{}".format(word,word_freq[word]))
    self.__stoi = defaultdict(lambda: 0)
    self.__stoi['<unk>'] = 0
    self.__stoi['<s>'] = 1
    self.__stoi['</s>'] = 2
    self.__itos = [''] * self.__size
    self.__itos[0] = '<unk>'
    self.__itos[1] = '<s>'
    self.__itos[2] = '</s>'
    for i, (k, v) in zip(range(self.__size - 3), sorted(word_freq.items(), key=lambda x: -x[1])):
      self.__stoi[k] = i + 3
      self.__itos[i + 3] = k
    # print(self.__itos)

    return self

  def save(self, filename):
    with codecs.open(filename,"w",encoding="utf-8") as fp:
      print(self.__size, file=fp)
      for i in range(self.__size):
        print(self.__itos[i], file=fp)

  @staticmethod
  def load(filename):
    with codecs.open(filename,"r",encoding="utf-8") as fp:
      self = Vocabulary()
      self.__size = int(next(fp))
      self.__stoi = defaultdict(lambda: 0)
      self.__itos = [''] * self.__size
      for i in range(self.__size):
        s = next(fp).strip()
        if s:
          self.__stoi[s] = i
          self.__itos[i] = s
    
    return self

