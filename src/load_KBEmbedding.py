

import gzip
triple_path='/mounts/data/proj/wenpeng/Dataset/freebase/'


def load_triples(infile, line_no):
    #first load entity_vocab
    entity_file=open(triple_path+'entity_vocab_'+str(line_no)+'triples.txt', 'w')
    entity_vocab={}
#     for line in entity_file:
#         parts=line.strip().split('\t')
#         entity_vocab[parts[0]]=int(parts[1])
#     entity_file.close()

    #second load relation_vocab
    relation_file=open(triple_path+'relation_vocab_'+str(line_no)+'triples.txt', 'w')
    relation_vocab={}
    
#     for line in relation_file:
#         parts=line.strip().split('\t')
#         relation_vocab[parts[0]]=int(parts[1])
#     relation_file.close()
    entity_count=[]
    relation_count=[]
    #load triples
    line_control=line_no
    read_file=gzip.open(infile, 'r')
    line_co=0
    triples=[]
    for line in read_file:
        parts=line.strip().split('\t')
        head = parts[0]
        relation=parts[1]
        tail=parts[2]

        head_id=entity_vocab.get(head)
        if head_id is None:
            head_id=len(entity_vocab)
            entity_vocab[head]=head_id
            entity_file.write(head+'\t'+str(head_id)+'\n')
            entity_count.append(0) # if entity is head, do not count, just occupy a position
#         else:
#             entity_count[head_id]+=1
        
        relation_id=relation_vocab.get(relation)
        if relation_id is None:
            relation_id=len(relation_vocab)
            relation_vocab[relation]=relation_id
            relation_file.write(relation+'\t'+str(relation_id)+'\n')
            relation_count.append(1)  
        else:
            relation_count[relation_id]+=1          

        tail_id=entity_vocab.get(tail)
        if tail_id is None:
            tail_id=len(entity_vocab)
            entity_vocab[tail]=tail_id
            entity_file.write(tail+'\t'+str(tail_id)+'\n')
            entity_count.append(1)
        else:
            entity_count[tail_id]+=1
            
                        
        triples.append([head_id, relation_id, tail_id])
        line_co+=1
        if line_co==line_control:
            break
    #make zero entries in entity_count to be 1
    for i in range(len(entity_count)):
        if entity_count[i]==0:
            entity_count[i]=1
    read_file.close()
    entity_file.close()
    relation_file.close()
    return triples, len(entity_vocab), len(relation_vocab), entity_count, relation_count
        
# if __name__ == '__main__':
#     vocabulize_triples(triple_path+'triples.txt.gz')