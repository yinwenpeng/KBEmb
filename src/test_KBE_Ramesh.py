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
from load_KBEmbedding import load_triples, load_TrainDevTest_triples_RankingLoss
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import create_nGRUs_para, one_iteration, load_model_from_file, store_model_to_file, norm_matrix, GRU_Combine_2Vector, get_negas, GRU_Combine_2Matrix, create_nGRUs_para_Ramesh, one_iteration_parallel, create_GRU_para, one_batch_parallel, all_batches, GRU_forward_one_triple, GRU_Combine_2Vector
from random import shuffle

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

from scipy import linalg, mat, dot

# from preprocess_wikiQA import compute_map_mrr

#need to change
'''

1, 94.561-->99.387@1000 --> 63.19
2, 94.608-->99.3871@1000--> 63.469
3, 94.6067->99.3888@1000--> 63.452
4, 94.654-->99.3905@1000
'''

def evaluate_lenet5(learning_rate=0.08, n_epochs=2000, nkerns=[50], batch_size=1000, window_width=4,
                    maxSentLength=64, emb_size=50, hidden_size=50,
                    margin=0.5, L2_weight=0.0004, update_freq=1, norm_threshold=5.0, max_truncate=40, line_no=483142, comment='v5_margin0.6_neg300_'):
    maxSentLength=max_truncate+2*(window_width-1)
    model_options = locals().copy()
    print "model options", model_options
    triple_path='/mounts/data/proj/wenpeng/Dataset/freebase/FB15k/'
    rng = numpy.random.RandomState(1234)
    triples, entity_size, relation_size, train_triples_set, train_entity_set, train_relation_set,dev_triples, dev_triples_set, dev_entity_set, dev_relation_set, test_triples, test_triples_set, test_entity_set, test_relation_set=load_TrainDevTest_triples_RankingLoss(triple_path+'freebase_mtr100_mte100-train.txt',triple_path+'freebase_mtr100_mte100-valid.txt', triple_path+'freebase_mtr100_mte100-test.txt', line_no, triple_path)
    
    
    print 'triple size:', len(triples), 'entity_size:', entity_size, 'relation_size:', relation_size#, len(entity_count), len(relation_count)
    dev_size=len(dev_triples)
    print 'dev triple size:', dev_size, 'entity_size:', len(dev_entity_set)
    test_size=len(test_triples)
    print 'test triple size:', test_size, 'entity_size:', len(test_entity_set)
#     print triples
#     print entity_count
#     print relation_count
#     exit(0)
    #datasets, vocab_size=load_wikiQA_corpus(rootPath+'vocab_lower_in_word2vec.txt', rootPath+'WikiQA-train.txt', rootPath+'test_filtered.txt', maxSentLength)#vocab_size contain train, dev and test
#     mtPath='/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST/'
#     mt_train, mt_test=load_mts_wikiQA(mtPath+'result_train/concate_2mt_train.txt', mtPath+'result_test/concate_2mt_test.txt')
#     wm_train, wm_test=load_wmf_wikiQA(rootPath+'train_word_matching_scores.txt', rootPath+'test_word_matching_scores.txt')
    #wm_train, wm_test=load_wmf_wikiQA(rootPath+'train_word_matching_scores_normalized.txt', rootPath+'test_word_matching_scores_normalized.txt')

    
#     entity_count=theano.shared(numpy.asarray(entity_count, dtype=theano.config.floatX), borrow=True)
#     entity_count=T.cast(entity_count, 'int64')
#     relation_count=theano.shared(numpy.asarray(relation_count, dtype=theano.config.floatX), borrow=True)
#     relation_count=T.cast(relation_count, 'int64')    


    rand_values=random_value_normal((entity_size, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
    entity_E=theano.shared(value=rand_values, borrow=True)      
    rand_values=random_value_normal((relation_size, emb_size), theano.config.floatX, numpy.random.RandomState(4321))
    relation_E=theano.shared(value=rand_values, borrow=True)    
    
    GRU_U, GRU_W, GRU_b=create_GRU_para(rng, word_dim=emb_size, hidden_dim=emb_size)  
#     GRU_U_combine, GRU_W_combine, GRU_b_combine=create_nGRUs_para(rng, word_dim=emb_size, hidden_dim=emb_size, n=3) 
    
    para_to_load=[entity_E, relation_E, GRU_U, GRU_W, GRU_b]
    load_model_from_file(triple_path+comment+'Best_Paras_dim'+str(emb_size), para_to_load)
    norm_entity_E=norm_matrix(entity_E)
    norm_relation_E=norm_matrix(relation_E)
    
    n_batchs=line_no/batch_size
    remain_triples=line_no%batch_size
    if remain_triples>0:
        batch_start=list(numpy.arange(n_batchs)*batch_size)+[line_no-batch_size]
    else:
        batch_start=list(numpy.arange(n_batchs)*batch_size)
    batch_start=theano.shared(numpy.asarray(batch_start, dtype=theano.config.floatX), borrow=True)
    batch_start=T.cast(batch_start, 'int64')   
    

    test_triple = T.lvector('test_triple')  
    neg_inds = T.lvector('neg_inds')

    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    predicted_tail=GRU_Combine_2Vector(norm_entity_E[test_triple[0]], norm_relation_E[test_triple[1]], emb_size, GRU_U, GRU_W, GRU_b)
    golden_tail=norm_entity_E[test_triple[2]]
    pos_loss=(1-cosine(predicted_tail,golden_tail))**2
    neg_Es=norm_entity_E[neg_inds].reshape((neg_inds.shape[0], emb_size))
    predicted_tail=predicted_tail.reshape((1, emb_size))
    multi=T.sum(predicted_tail*neg_Es, axis=1)
    len1=T.sqrt(T.sum(predicted_tail**2))
    len2=T.sqrt(T.sum(neg_Es**2, axis=1))
    cos=multi/(len1*len2)
    neg_loss_vector=(1-cos)**2

#     normed_predicted_tail=predicted_tail/T.sqrt(T.sum(predicted_tail**2))
#     
#     pos_loss=T.sum(abs(normed_predicted_tail-golden_tail))
#     neg_Es=norm_entity_E[neg_inds].reshape((neg_inds.shape[0], emb_size))
#     predicted_tail=normed_predicted_tail.reshape((1, emb_size))
# 
#     neg_loss_vector=T.sum(abs(predicted_tail-neg_Es), axis=1)
   
    
    
    
    GRU_forward_step = theano.function([test_triple, neg_inds], [pos_loss,neg_loss_vector], on_unused_input='ignore')
    
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
    corpus_triples_set=train_triples_set|dev_triples_set|test_triples_set
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0
        #shuffle(train_batch_start)#shuffle training data
#         cost_1, cost_l= train_model(triples)
#                 #print 'layer3_input', layer3_input
#         print 'cost:', cost_1, cost_l
        
        #test
        test_size=len(test_triples)
        hits_10=test_size
        hits_1=test_size
        
        co=0
        for test_triple in test_triples:
            co+=1

            count=0
            flag_continue=True
            nega_entity_set=get_negas(test_triple, corpus_triples_set, test_entity_set)
#             print len(nega_entity_set)
            p_loss, n_loss_vector=GRU_forward_step(test_triple, list(nega_entity_set))

            n_loss_vector=numpy.sort(n_loss_vector)
#             print p_loss
#             print n_loss_vector[:15]
#             exit(0)
            if p_loss>n_loss_vector[0]:
                hits_1-=1
            if p_loss>n_loss_vector[9]:
                hits_10-=1 
            if co%1000==0:
                print co, '...'
                print '\t\thits_10', hits_10*100.0/test_size, 'hits_1', hits_1*100.0/test_size
        hits_10=hits_10*100.0/test_size
        hits_1=hits_1*100.0/test_size
        
        
#             if patience <= iter:
#                 done_looping = True
#                 break
        #after each epoch, increase the batch_size
        store_model_to_file(triple_path+'Best_Paras_dim'+str(emb_size)+'_hits10_'+str(hits_10)[:6], para_to_load)
        print 'Finished storing best  params'
        print 'Epoch ', epoch, 'uses ', (time.clock()-mid_time)/60.0, 'min, Hits_10:',  hits_10, 'Hits_1:,', hits_1
        mid_time = time.clock()
        exit(0)
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




def cosine(vec1, vec2):
    vec1=debug_print(vec1, 'vec1')
    vec2=debug_print(vec2, 'vec2')
    norm_uni_l=T.sqrt((vec1**2).sum())
    norm_uni_r=T.sqrt((vec2**2).sum())
    
    dot=T.dot(vec1,vec2.T)
    
    simi=debug_print(dot/(norm_uni_l*norm_uni_r), 'uni-cosine')
    return simi   
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