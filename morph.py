from keras.layers import Input, Dense, SimpleRNN, RepeatVector, TimeDistributed, Masking
from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy

import plotly.graph_objs as go
import plotly

DATA = '/home/zakirov/Downloads/zdf.txt'

# setup
MSIZE   = 40#100
SEQSIZE = 20
VSPLIT  = 0.1

# useful funcs
def getword(x, id2char):
    assert len(x.shape) == 2 and x.shape[0] == SEQSIZE and x.shape[1] == len(id2char)
    word = ''
    for i in range(x.shape[0]):
        if numpy.sum(x[i]) == 0: break
        c = id2char[numpy.argmax(x[i])]
        if c == '\n': break
        word += c
    return word

def word2data(words, char2id):
    data = numpy.zeros(shape=(len(words), SEQSIZE, CSIZE), dtype='int8')
    for i, word in enumerate(words):
        for j, c in enumerate(word[:SEQSIZE]):
            data[i][j][char2id[c]] = 1
    return data

# load data
char2id = dict([(c, i) for i, c in enumerate(sorted(list(set(open(DATA).read()))))])
id2char = dict([(i, c) for c, i in char2id.items()])

CSIZE   = len(char2id)
all_words   = open(DATA).readlines()
SIZE    = len(all_words)
all_data = numpy.zeros(shape=(SIZE, SEQSIZE, CSIZE), dtype='int8')
for i, word in enumerate(all_words):
    for j, c in enumerate(word[:SEQSIZE]):
        all_data[i][j][char2id[c]] = 1
numpy.random.shuffle(all_data)
vl_data = all_data[:int(VSPLIT * len(all_data))]
tr_data = all_data[int(VSPLIT * len(all_data)):]

# build model
inp = Input(shape=(SEQSIZE, CSIZE))
x = Masking()(inp)
x = SimpleRNN(units=MSIZE)(x)
encoder = Model(inputs=inp, outputs=x)
encoder.compile(optimizer='sgd', loss='mse')
x = RepeatVector(SEQSIZE)(x)
dewg  = SimpleRNN(units=MSIZE, return_sequences=True)
dewg2 = Dense(CSIZE, activation='softmax')
x = dewg(x)
x = TimeDistributed(dewg2)(x)

model = Model(inputs=inp, outputs=x)
model.compile(optimizer=adam(lr=0.001), loss='categorical_crossentropy')

inpdec = Input(shape=(MSIZE, ))
x = RepeatVector(SEQSIZE)(inpdec)
x = dewg(x)
x = TimeDistributed(dewg2)(x)
decoder = Model(inputs=inpdec, outputs=x)
decoder.compile(optimizer='sgd', loss='mse')

def LrDecay(epoch):
    if epoch > 70:
        return 0.0001
    else:
        return 0.001

if 0:
    clbck = [ModelCheckpoint('wg_40/weights.{epoch:03d}-{val_loss:.2f}.hdf5'), LearningRateScheduler(LrDecay)]
    model.fit(x=tr_data, y=tr_data, validation_data=(vl_data, vl_data), epochs=200, callbacks=clbck, initial_epoch=0)
    if 1:
        l_data     = numpy.array(model.history.history['loss'])
        val_l_data = numpy.array(model.history.history['val_loss'])
        scat_l      = go.Scatter(x=numpy.array(range(len(l_data))), y=l_data,     mode='lines', name='learn')
        scat_val_l  = go.Scatter(x=numpy.array(range(len(l_data))), y=val_l_data, mode='lines', name='validation')
        plotly.offline.plot(go.Figure(data=[scat_l, scat_val_l],
                                      layout=go.Layout(yaxis=dict(title='error'),
                                                       xaxis=dict(title='epoch'))))
else:
    model.load_weights('wg_40/weights.199-0.15.hdf5')#'wg_100/weights.049-0.02.hdf5')

def SelfCheck(model, encoder, decoder, data2test):
    data2test_rec = model.predict(x=data2test)
    en_rec = encoder.predict(x=data2test)
    dec_rec = decoder.predict(x=en_rec)
    if numpy.sum(data2test_rec - dec_rec) != 0:
        assert 0

def MeasureRecon(model, data2test, id2char):
    data2test_rec = model.predict(x=data2test)
    err_rate = 0
    for i in range(len(data2test)):
        word     = getword(data2test[i], id2char)
        rec_word = getword(data2test_rec[i], id2char)
        if word != rec_word:
            print(word, '<>', rec_word)
            err_rate += 1
    print('Error rate: ', 100 * float(err_rate) / len(data2test), '%')

def ConstructReconstruct(word, model, char2id, id2char):
    data = numpy.zeros(shape=(1, SEQSIZE, CSIZE), dtype='int8')
    for j, c in enumerate(word[:SEQSIZE]):
        data[0][j][char2id[c]] = 1
    data_rec = model.predict(x=data)
    rec_word = getword(data_rec[0], id2char)
    print(word, '<>', rec_word)

def ConstructWords(encoder, decoder, words, id2char, char2id):
    for word1, word2 in words:
        w1 = word2data([word1], char2id)
        w2 = word2data([word2], char2id)
        e_w1 = encoder.predict(x=w1)
        e_w2 = encoder.predict(x=w2)
        e_w1w2 = (e_w1 + e_w2) / 2
        r_w1w2 = decoder.predict(x=e_w1w2)
        wordr = getword(r_w1w2[0], id2char)
        print(word1, word2, wordr)

def NormVector(X):
    try:
        return (X.T / numpy.linalg.norm(X, ord=2, axis=1)).T
    except:
        pass

def WordAnalogy(encoder, decoder, words, id2char, char2id):
    for word1, word2, word3 in words:
        w1 = word2data([word1], char2id)
        w2 = word2data([word2], char2id)
        w3 = word2data([word3], char2id)
        e_w1 = encoder.predict(x=w1)
        e_w2 = encoder.predict(x=w2)
        e_w3 = encoder.predict(x=w3)
        e_w1w2 = NormVector(NormVector(NormVector(e_w1) - NormVector(e_w2)) + NormVector(e_w3))
        r_w1w2 = decoder.predict(x=e_w1w2)
        wordr = getword(r_w1w2[0], id2char)
        print(word1, '-', word2, '+', word3, '=',wordr)

def FindNearest(encoder, words, all_words, char2id):
    A = NormVector(encoder.predict(word2data(words, char2id)))
    B = NormVector(encoder.predict(word2data(all_words, char2id)))
    R = numpy.dot(A, B.T)
    I = numpy.flip(numpy.argsort(R, axis=1), axis=1)
    for i, word in enumerate(words):
        print(word.rstrip())
        for k in range(10):
            print('\t', R[i][I[i]][k], all_words[I[i][k]].rstrip())

SelfCheck(model, encoder, decoder, tr_data[0 : 1000])
FindNearest(encoder, ['купить\n', 'лить\n', 'дать\n', 'катить\n'], all_words, char2id)

ConstructReconstruct('глокая\n', model, char2id, id2char)
ConstructReconstruct('куздра\n', model, char2id, id2char)
ConstructReconstruct('быдланула\n', model, char2id, id2char)
ConstructReconstruct('штеко\n', model, char2id, id2char)
ConstructReconstruct('бокренка\n', model, char2id, id2char)

ConstructReconstruct('кагоцел\n', model, char2id, id2char)
ConstructReconstruct('живой-журнал\n', model, char2id, id2char)
ConstructReconstruct('фсбшник\n', model, char2id, id2char)
ConstructReconstruct('абвгдейка\n', model, char2id, id2char)
ConstructReconstruct('громозавр\n', model, char2id, id2char)

wordpairs = [('хлеб\n','завод\n'), ('великий\n','луг\n'), ('пере\n','ехать\n'),
             ('судно\n', 'строитель\n'), ('судо\n','строитель\n'), ('при\n','ехать\n'), ]

word3s = [('приехать\n','ехать\n','быть\n')]

ConstructWords(encoder, decoder, wordpairs, id2char, char2id)
WordAnalogy(encoder, decoder, word3s, id2char, char2id)

print('TRAIN')
MeasureRecon(model, tr_data[0 : 1000], id2char)
print('VALIDATION')
MeasureRecon(model, vl_data[0 : 1000], id2char)
