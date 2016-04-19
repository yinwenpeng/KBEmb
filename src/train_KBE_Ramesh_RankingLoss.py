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
from common_functions import create_nGRUs_para, get_n_neg_triples_new, norm_matrix, load_model_from_file, one_batch_parallel_Ramesh, one_neg_batches_parallel_Ramesh, GRU_Combine_2Matrix, create_nGRUs_para_Ramesh, one_iteration_parallel, create_GRU_para, one_batch_parallel, all_batches, store_model_to_file

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

def evaluate_lenet5(learning_rate=0.5, n_epochs=2000, nkerns=[50], batch_size=1000, window_width=4,
                    maxSentLength=64, emb_size=50, hidden_size=50,
                    margin=0.25, L2_weight=1e-10, update_freq=1, norm_threshold=5.0, max_truncate=40, line_no=483142, neg_size=60):
    maxSentLength=max_truncate+2*(window_width-1)
    model_options = locals().copy()
    print "model options", model_options
    triple_path='/mounts/data/proj/wenpeng/Dataset/freebase/FB15k/'
    rng = numpy.random.RandomState(1234)
#     triples, entity_size, relation_size, entity_count, relation_count=load_triples(triple_path+'freebase_mtr100_mte100-train.txt', line_no, triple_path)#vocab_size contain train, dev and test
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
    load_model_from_file(triple_path+'Best_Paras_dim'+str(emb_size), para_to_load)
    norm_entity_E=norm_matrix(entity_E)
    norm_relation_E=norm_matrix(relation_E)
    
        
    n_batchs=line_no/batch_size
    remain_triples=line_no%batch_size
    if remain_triples>0:
        batch_start=list(numpy.arange(n_batchs)*batch_size)+[line_no-batch_size]
    else:
        batch_start=list(numpy.arange(n_batchs)*batch_size)

    n_batchs_test=test_size/batch_size
    remain_triples_test=test_size%batch_size
    if remain_triples_test>0:
        batch_start_test=list(numpy.arange(n_batchs_test)*batch_size)+[test_size-batch_size]
    else:
        batch_start_test=list(numpy.arange(n_batchs_test)*batch_size)
#     batch_start=theano.shared(numpy.asarray(batch_start, dtype=theano.config.floatX), borrow=True)
#     batch_start=T.cast(batch_start, 'int64')   
    
    # allocate symbolic variables for the data
#     index = T.lscalar()
    x_index_l = T.lmatrix('x_index_l')   # now, x is the index matrix, must be integer
    n_index_T = T.ltensor3('n_index_T')
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
    

    
    dist_tail=one_batch_parallel_Ramesh(x_index_l, norm_entity_E, norm_relation_E, GRU_U, GRU_W, GRU_b, emb_size)
    
    
    loss__tail_is=one_neg_batches_parallel_Ramesh(n_index_T, norm_entity_E, norm_relation_E, GRU_U, GRU_W, GRU_b, emb_size)
    loss_tail_i=T.maximum(0.0, margin+dist_tail.reshape((dist_tail.shape[0],1))-loss__tail_is) 
    loss_tail_i_test=T.maximum(0.0, 0.0+dist_tail.reshape((dist_tail.shape[0],1))-loss__tail_is)   
    binary_matrix_test=T.gt(loss_tail_i_test, 0)
    sum_vector_test=T.sum(binary_matrix_test, axis=1)
    binary_vector_hits10=T.gt(sum_vector_test, 10)
    test_loss=T.sum(binary_vector_hits10)*1.0/batch_size  
#     loss_relation_i=T.maximum(0.0, margin+dis_relation.reshape((dis_relation.shape[0],1))-loss__relation_is) 
#     loss_head_i=T.maximum(0.0, margin+dis_head.reshape((dis_head.shape[0],1))-loss__head_is)     
#     def neg_slice(neg_matrix):
#         dist_tail_slice, dis_relation_slice, dis_head_slice=one_batch_parallel_Ramesh(neg_matrix, entity_E, relation_E, GRU_U_combine, GRU_W_combine, GRU_b_combine, emb_size)
#         loss_tail_i=T.maximum(0.0, margin+dist_tail-dist_tail_slice) 
#         loss_relation_i=T.maximum(0.0, margin+dis_relation-dis_relation_slice) 
#         loss_head_i=T.maximum(0.0, margin+dis_head-dis_head_slice) 
#         return loss_tail_i, loss_relation_i, loss_head_i
#     
#     (loss__tail_is, loss__relation_is, loss__head_is), updates = theano.scan(
#        neg_slice,
#        sequences=n_index_T,
#        outputs_info=None)  
    
    loss_tails=T.mean(T.sum(loss_tail_i, axis=1) )
#     loss_relations=T.mean(T.sum(loss_relation_i, axis=1) )
#     loss_heads=T.mean(T.sum(loss_head_i, axis=1) )
    loss=loss_tails#+loss_relations+loss_heads
    L2_loss=debug_print((entity_E** 2).sum()+(relation_E** 2).sum()\
                      +(GRU_U** 2).sum()+(GRU_W** 2).sum(), 'L2_reg')
    cost=loss+L2_weight*L2_loss
    #params = layer3.params + layer2.params + layer1.params+ [conv_W, conv_b]
    params = [entity_E, relation_E, GRU_U, GRU_W, GRU_b]
#     params_conv = [conv_W, conv_b]
    params_to_store=[entity_E, relation_E, GRU_U, GRU_W, GRU_b]
    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
#         grad_i=debug_print(grad_i,'grad_i')
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc+1e-9)))   #AdaGrad
        updates.append((acc_i, acc))    
  
    train_model = theano.function([x_index_l, n_index_T], [loss, cost], updates=updates,on_unused_input='ignore')
    test_model = theano.function([x_index_l, n_index_T], test_loss, on_unused_input='ignore')
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
    
    best_test_loss=1000000
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0
        #shuffle(train_batch_start)#shuffle training data
        for start in batch_start:
            if start%100000==0:
                print start, '...'
            pos_triples=triples[start:start+batch_size]
            all_negs=[]
            for pos_triple in pos_triples:
                neg_triples=get_n_neg_triples_new(pos_triple, train_triples_set, train_entity_set, train_relation_set, neg_size/2, True)
# #                 print 'neg_head_triples'
#                 neg_relation_triples=get_n_neg_triples(pos_triple, train_triples_set, train_entity_set, train_relation_set, 1, neg_size/3)
# #                 print 'neg_relation_triples'
#                 neg_tail_triples=get_n_neg_triples(pos_triple, train_triples_set, train_entity_set, train_relation_set, 2, neg_size/3)
#                 print 'neg_tail_triples'
                all_negs.append(neg_triples)
                
            neg_tensor=numpy.asarray(all_negs).reshape((batch_size, neg_size, 3)).transpose(1,0,2)
            loss, cost= train_model(pos_triples, neg_tensor)
        print 'Training loss:', loss, 'cost:', cost
        
        
        loss_test=0.0
        corpus_triples_set=train_triples_set|dev_triples_set|test_triples_set
        for test_start in batch_start_test:
            pos_triples=test_triples[test_start:test_start+batch_size]
            all_negs=[]
            for pos_triple in pos_triples:
                neg_triples=get_n_neg_triples_new(pos_triple, corpus_triples_set, test_entity_set, test_relation_set, neg_size/2, True)
                all_negs.append(neg_triples)
                
            neg_tensor=numpy.asarray(all_negs).reshape((batch_size, neg_size, 3)).transpose(1,0,2)
            loss_test+= test_model(pos_triples, neg_tensor)
            
            
        loss_test/=n_batchs_test
        print '\t\t\tUpdating epoch', epoch, 'finished! Test hits10:', 1.0-loss_test
        if loss_test< best_test_loss:
            store_model_to_file(triple_path+'Best_Paras_dim'+str(emb_size), params_to_store)
            best_test_loss=loss_test
            print 'Finished storing best  params'
#             exit(0)
        print 'Epoch ', epoch, 'uses ', (time.clock()-mid_time)/60.0, 'min'
        mid_time = time.clock()            
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