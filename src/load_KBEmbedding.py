

import gzip
triple_path='/mounts/data/proj/wenpeng/Dataset/freebase/'


def load_triples(infile, line_no):
    #first load entity_vocab
    entity_file=open(triple_path+'entity_vocab.txt', 'r')
    entity_vocab={}
    for line in entity_file:
        parts=line.strip().split('\t')
        entity_vocab[parts[0]]=int(parts[1])
    entity_file.close()
    #second load relation_vocab
    relation_file=open(triple_path+'relation_vocab.txt', 'r')
    relation_vocab={}
    for line in relation_file:
        parts=line.strip().split('\t')
        relation_vocab[parts[0]]=int(parts[1])
    relation_file.close()
    #load triples
    line_control=line_no
    read_file=gzip.open(infile, 'r')
    line_co=0
    triples=[]
    for line in read_file:
        parts=line.strip().split('\t')
        triples.append([entity_vocab.get(parts[0]), relation_vocab.get(parts[1]), entity_vocab.get(parts[2])])
        line_co+=1
        if line_co==line_control:
            break
    read_file.close()
    return triples, len(entity_vocab), len(relation_vocab)
        
# if __name__ == '__main__':
#     vocabulize_triples(triple_path+'triples.txt.gz')