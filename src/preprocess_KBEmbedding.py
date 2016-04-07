import gzip
triple_path='/mounts/data/proj/wenpeng/Dataset/freebase/'


def vocabulize_triples(infile):
    line_control=1000
    read_file=gzip.open(infile, 'r')
    write_entity=open(triple_path+'entity_vocab.txt', 'w')
    write_relation=open(triple_path+'relation_vocab.txt', 'w')
    line_co=0
    entity_vocab=set()
    relation_vocab=set()
    for line in read_file:
        parts=line.strip().split('\t')
        if len(parts)!=3:
            print 'incomplete triple:', line
            exit(0)
        else:
            if parts[0] not in entity_vocab:
                entity_vocab.add(parts[0])

            if parts[1] not in relation_vocab:
                relation_vocab.add(parts[1])

            if parts[2] not in entity_vocab:
                entity_vocab.add(parts[2])      
            line_co+=1
            if line_co==line_control:
                break
    read_file.close()
    co=0
    for entity in entity_vocab:
        write_entity.write(entity+'\t'+str(co)+'\n')
        co+=1
    co=0
    for relation in relation_vocab:
        write_relation.write(relation+'\t'+str(co)+'\n')
        co+=1
    write_entity.close()
    write_relation.close()  
    print 'finished'
if __name__ == '__main__':
    vocabulize_triples(triple_path+'triples.txt.gz')