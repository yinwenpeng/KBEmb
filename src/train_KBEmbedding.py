import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN
import time

# from logistic_sgd import LogisticRegression
# from mlp import HiddenLayer
# from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from load_KBEmbedding import load_triples
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import create_nGRUs_para, one_iteration, GRU_Combine_2Matrix, create_nGRUs_para_Ramesh, one_iteration_parallel, create_GRU_para
from random import shuffle

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

from scipy import linalg, mat, dot

# from preprocess_wikiQA import compute_map_mrr

#need to change
'''


4) fine-tune word embeddings
5) translation bettween
6) max sentence length to 40:   good and best
7) implement attention by euclid, not cosine: good
8) stop words by Yi Yang
9) normalized first word matching feature
10) only use bleu1 and nist1
11) only use bleu4 and nist5



Doesnt work:
1) lr0.08, kern30, window=5, update10
8) kern as Yu's paper
7) shuffle training data: should influence little as batch size is 1
3) use bleu and nist scores
1) true sentence lengths
2) unnormalized sentence length
8) euclid uses 1/exp(x)
'''

def evaluate_lenet5(learning_rate=0.08, n_epochs=2000, nkerns=[50], batch_size=1, window_width=4,
                    maxSentLength=64, emb_size=5, hidden_size=200,
                    margin=0.5, L2_weight=0.0004, update_freq=1, norm_threshold=5.0, max_truncate=40):
    maxSentLength=max_truncate+2*(window_width-1)
    model_options = locals().copy()
    print "model options", model_options
    triple_path='/mounts/data/proj/wenpeng/Dataset/freebase/'
    rng = numpy.random.RandomState(1234)
    triples, entity_size, relation_size, entity_count, relation_count=load_triples(triple_path+'triples.txt.gz', line_no=1000)#vocab_size contain train, dev and test
    print 'triple size:', len(triples), 'entity_size:', entity_size, 'relation_size:', relation_size#, len(entity_count), len(relation_count)
#     print triples
#     print entity_count
#     print relation_count
#     exit(0)
    #datasets, vocab_size=load_wikiQA_corpus(rootPath+'vocab_lower_in_word2vec.txt', rootPath+'WikiQA-train.txt', rootPath+'test_filtered.txt', maxSentLength)#vocab_size contain train, dev and test
#     mtPath='/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST/'
#     mt_train, mt_test=load_mts_wikiQA(mtPath+'result_train/concate_2mt_train.txt', mtPath+'result_test/concate_2mt_test.txt')
#     wm_train, wm_test=load_wmf_wikiQA(rootPath+'train_word_matching_scores.txt', rootPath+'test_word_matching_scores.txt')
    #wm_train, wm_test=load_wmf_wikiQA(rootPath+'train_word_matching_scores_normalized.txt', rootPath+'test_word_matching_scores_normalized.txt')

    
    entity_count=theano.shared(numpy.asarray(entity_count, dtype=theano.config.floatX), borrow=True)
    entity_count=T.cast(entity_count, 'int64')
    relation_count=theano.shared(numpy.asarray(relation_count, dtype=theano.config.floatX), borrow=True)
    relation_count=T.cast(relation_count, 'int64')    


    rand_values=random_value_normal((entity_size, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
    entity_E=theano.shared(value=rand_values, borrow=True)      
    rand_values=random_value_normal((relation_size, emb_size), theano.config.floatX, numpy.random.RandomState(4321))
    relation_E=theano.shared(value=rand_values, borrow=True)    
    
    GRU_U, GRU_W, GRU_b=create_GRU_para(rng, word_dim=emb_size, hidden_dim=emb_size)  
    GRU_U_combine, GRU_W_combine, GRU_b_combine=create_nGRUs_para(rng, word_dim=emb_size, hidden_dim=emb_size, n=2) 
    #cost_tmp=0
    error_sum=0
    
    # allocate symbolic variables for the data
#     index = T.lscalar()
    x_index_l = T.lmatrix('x_index_l')   # now, x is the index matrix, must be integer
#     x_index_r = T.imatrix('x_index_r')
#     y = T.ivector('y')  
#     left_l=T.iscalar()
#     right_l=T.iscalar()
#     left_r=T.iscalar()
#     right_r=T.iscalar()
#     length_l=T.iscalar()
#     length_r=T.iscalar()
#     norm_length_l=T.fscalar()
#     norm_length_r=T.fscalar()
#     mts=T.fmatrix()
#     wmf=T.fmatrix()
#     cost_tmp=T.fscalar()
#     #x=embeddings[x_index.flatten()].reshape(((batch_size*4),maxSentLength, emb_size)).transpose(0, 2, 1).flatten()
#     ishape = (emb_size, maxSentLength)  # this is the size of MNIST images
#     filter_size=(emb_size,window_width)
#     #poolsize1=(1, ishape[1]-filter_size[1]+1) #?????????????????????????????
#     length_after_wideConv=ishape[1]+filter_size[1]-1
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    #layer0_input = x.reshape(((batch_size*4), 1, ishape[0], ishape[1]))

    #load embedding, scan for each triple, run GRU, generate new embedding matrix, return
    
    entity_E_hat_1, relation_E_hat_1=one_iteration_parallel(x_index_l, entity_E, relation_E, GRU_U, GRU_W, GRU_b, emb_size, entity_size, relation_size, entity_count, relation_count)     
    entity_E_updated_1=GRU_Combine_2Matrix(entity_E, entity_E_hat_1, emb_size, GRU_U_combine[0], GRU_W_combine[0], GRU_b_combine[0])
    relation_E_updated_1=GRU_Combine_2Matrix(relation_E, relation_E_hat_1, emb_size, GRU_U_combine[1], GRU_W_combine[1], GRU_b_combine[1])
#     cost=((entity_E_hat_1-entity_E)**2).sum()+((relation_E_hat_1-relation_E)**2).sum()
    cost_1=((entity_E_updated_1-entity_E)**2).sum()+((relation_E_updated_1-relation_E)**2).sum()
    
    entity_E_hat_2, relation_E_hat_2=one_iteration_parallel(x_index_l, entity_E_updated_1, relation_E_updated_1, GRU_U, GRU_W, GRU_b, emb_size, entity_size, relation_size, entity_count, relation_count)
    entity_E_last_2=GRU_Combine_2Matrix(entity_E_updated_1, entity_E_hat_2, emb_size, GRU_U_combine[0], GRU_W_combine[0], GRU_b_combine[0])
    relation_E_last_2=GRU_Combine_2Matrix(relation_E_updated_1, relation_E_hat_2, emb_size, GRU_U_combine[1], GRU_W_combine[1], GRU_b_combine[1])    
     
    L2_loss=debug_print((entity_E** 2).sum()+(relation_E** 2).sum()\
                      +(GRU_U** 2).sum()+(GRU_W** 2).sum()\
                      +(GRU_U_combine** 2).sum()+(GRU_W_combine** 2).sum(), 'L2_reg')
    cost_sys=((entity_E_last_2-entity_E_updated_1)**2).sum()+((relation_E_last_2-relation_E_updated_1)**2).sum()
    cost=cost_sys+L2_weight*L2_loss
    #params = layer3.params + layer2.params + layer1.params+ [conv_W, conv_b]
    params = [entity_E, relation_E, GRU_U, GRU_W, GRU_b, GRU_U_combine, GRU_W_combine, GRU_b_combine]
#     params_conv = [conv_W, conv_b]
    
    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        grad_i=debug_print(grad_i,'grad_i')
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
        updates.append((acc_i, acc))    
  
    train_model = theano.function([x_index_l], [cost_1,cost_sys], updates=updates,on_unused_input='ignore')
# 
#     train_model_predict = theano.function([index], [cost_this,layer3.errors(y), layer3_input, y],
#           givens={
#             x_index_l: indices_train_l[index: index + batch_size],
#             x_index_r: indices_train_r[index: index + batch_size],
#             y: trainY[index: index + batch_size],
#             left_l: trainLeftPad_l[index],
#             right_l: trainRightPad_l[index],
#             left_r: trainLeftPad_r[index],
#             right_r: trainRightPad_r[index],
#             length_l: trainLengths_l[index],
#             length_r: trainLengths_r[index],
#             norm_length_l: normalized_train_length_l[index],
#             norm_length_r: normalized_train_length_r[index],
#             mts: mt_train[index: index + batch_size],
#             wmf: wm_train[index: index + batch_size]}, on_unused_input='ignore')



    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
#     validation_frequency = min(n_train_batches/5, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    mid_time = start_time

    epoch = 0
    done_looping = False
    
    svm_max=0.0
    best_epoch=0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0
        #shuffle(train_batch_start)#shuffle training data
        cost_1, cost_l= train_model(triples)
                #print 'layer3_input', layer3_input
        print 'cost:', cost_1, cost_l

#             if patience <= iter:
#                 done_looping = True
#                 break
        #after each epoch, increase the batch_size
        print 'Epoch ', epoch, 'uses ', (time.clock()-mid_time)/60.0, 'min'
        mid_time = time.clock()
#         exit(0)
        
#         #store the paras after epoch 15
#         if epoch ==22:
#             store_model_to_file(params_conv)
#             print 'Finished storing best conv params'
#             exit(0)
            
        #print 'Batch_size: ', update_freq
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def store_model_to_file(best_params):
    save_file = open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/Best_Conv_Para_ARCII', 'wb')  # this will overwrite current contents
    for para in best_params:           
        cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()

def cosine(vec1, vec2):
    vec1=debug_print(vec1, 'vec1')
    vec2=debug_print(vec2, 'vec2')
    norm_uni_l=T.sqrt((vec1**2).sum())
    norm_uni_r=T.sqrt((vec2**2).sum())
    
    dot=T.dot(vec1,vec2.T)
    
    simi=debug_print(dot/(norm_uni_l*norm_uni_r), 'uni-cosine')
    return simi.reshape((1,1))    
def Linear(sum_uni_l, sum_uni_r):
    return (T.dot(sum_uni_l,sum_uni_r.T)).reshape((1,1))    
def Poly(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    poly=(0.5*dot+1)**3
    return poly.reshape((1,1))    
def Sigmoid(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    return T.tanh(1.0*dot+1).reshape((1,1))    
def RBF(sum_uni_l, sum_uni_r):
    eucli=T.sum((sum_uni_l-sum_uni_r)**2)
    return T.exp(-0.5*eucli).reshape((1,1))    
def GESD (sum_uni_l, sum_uni_r):
    eucli=1/(1+T.sum((sum_uni_l-sum_uni_r)**2))
    kernel=1/(1+T.exp(-(T.dot(sum_uni_l,sum_uni_r.T)+1)))
    return (eucli*kernel).reshape((1,1))   
def EUCLID(sum_uni_l, sum_uni_r):
    return T.sqrt(T.sqr(sum_uni_l-sum_uni_r).sum()+1e-20).reshape((1,1))
    


if __name__ == '__main__':
    evaluate_lenet5()