''' 
To compile code, run python hmm_pos.py --train=pos.train.txt --test=sample_3.txt
You just need to pass the train text file to train and test text file to test.
'''
from __future__ import division
import argparse
from collections import defaultdict
import operator

parser = argparse.ArgumentParser(description='HMM-POS')
parser.add_argument('--train', type=str,
                   help='Train file input')
parser.add_argument('--test', type=str,
                   help='input test file')
                   
args = parser.parse_args()

train_corpus = args.train
test_file = args.test

print "\nViterbi Algorithm HMM Tagger by Muhammad Abdullah Jamal\n"

def lemma_input(token):
    if token[-4:] == 'sses' or token[-3:]=='xes':
        return token[:-2]
    elif token[-3:] == 'ses' or token[-3:]=='zes':
        return token[:-1]
    elif token[-4:] == 'ches' or token[-4:]=='shes':
        return token[:-2]
    elif token[-3:] == 'men':
        return token[:-2]+'an'
    elif token[-3:] == 'ies':
        return token[:-3]+'y'
    else:
        return token

def read_train(tr_file):
    token = []
    tag = []
    with open(tr_file) as f:
        for line in f:
            if line == '\n':
                continue
            else:
                tk = line[:-1].split(' ')
                token.append(lemma_input(tk[0].lower()))
                tag.append(tk[1])
            
    return token,tag
    
def sentences_tags_corps(tr_file):
    
    with open(tr_file, 'r') as corp:
        sentences = [] # sentences
        correctTags = [] # Correct tags
        tagData = [] # Sentences with word and tags

        words, tags, oneTag = [], [], []

        for line in corp:
            if line == '\n':
                correctTags.append(tags)
                sentences.append(words)
                tagData.append(oneTag)

                words, tags, oneTag  = [], [], []
            else:
                tk = line[:-1].split(' ')
                words.append(lemma_input(tk[0].lower()))
                tags.append(tk[1])
                bothTags = (lemma_input(tk[0].lower()), tk[1])
                oneTag.append(bothTags)
    return sentences, correctTags, tagData


def tags_observed(tag):
    unique_tag = list(set(tag))
    sort_tag = sorted(unique_tag)
    for i in range(0,len(sort_tag)):
        print str(i+1)+'  '+sort_tag[i]
    return sort_tag

def initial_dist(sort_tags,tags):
    init_dist = dict()
     
    for i in range(0,len(sort_tags)):
         count = 0
         for j in range(0,len(tags)):
              if sort_tags[i] == tags[j][0]:
                  count = count + 1
         init_dist[sort_tags[i]] = count/len(tags) 
         if count > 0:
            print "start ["+str(sort_tags[i])+" |  ] ", '{0:.6f}'.format(count/len(tags))   
    return init_dist

def cal_bigrams_lexicals(data_tuples):
    unigrams = dict() 
    bigrams = dict()
    lexicals = dict() 

    word = ''

    startTag = '' 
    endTag = ' ' 

    t2 = '' 
    t1 = startTag 
    t0 = '' 

    for sent in data_tuples:
      
        sentence = sent
        sentence.append(('',''))

        for tagWord in sentence:
            
            if tagWord[1] == '': 
               
                unigrams[startTag] = unigrams.get(startTag, 0) + 1
                word, t0 = '', endTag
            else:
                word, t0 = tagWord
                
            bigram = ' '.join([t1, t0]) # Create bigram 
            bigrams[bigram] = bigrams.get(bigram, 0) + 1 
            unigrams[t0] = unigrams.get(t0, 0) + 1 
            
            lexicals[word] = lexicals.get(word, {t0 : 0}) 
            lexicals[word][t0] = lexicals[word].get(t0, 0) + 1 

            
            if word == '': 
               t2 = ''
               t1 = startTag
            else:
               t2 = t1
               t1 = t0

    unigram = dict()
    bigram = dict()
    lexical = dict()

    for uni in unigrams:
        unigram[uni] = unigrams[uni] / sum(unigrams.values())

    for bi in bigrams:
        bigram[bi] = bigrams[bi] / unigrams[bi.split(' ')[0]]

    for word in lexicals:
        lexical[word] = {}
        for tag in lexicals[word]:
            lexical[word][tag] = lexicals[word][tag] / sum(lexicals[word].values())

    return unigram, bigram, lexical

def emission_prob(token_words,token_tags):
    		
    e_values_c = dict()
    tag_c = dict()
    for word, tag in zip(token_words, token_tags):
       if (word, tag) not in e_values_c:
           e_values_c[(word, tag)] = 1
       else:
           e_values_c[(word, tag)] += 1
       if tag not in tag_c:
           tag_c[tag] = 1
       else:
           tag_c[tag] +=1

    e_values = {(word, tag): (e_values_c[(word, tag)])/(tag_c[tag]) for word, tag in e_values_c}
  
    taglist = set(tag_c)

    emissions = e_values.keys()
    emissions = sorted(emissions, key=lambda x: x[0])
    for item in emissions:
     print("{}  {} {:0.6f}".format(item[0], item[1], e_values[item]))
    
    return e_values
    

def print_transition_prob(prob_dict,unigram):
    
    for tags in unigram:
        if tags!='``':
            print("[ {:0.6f} ]".format(1.00000)),
            for tags_ in unigram:
                newstr = str(tags)+' '+str(tags_)
                if newstr in prob_dict:
                    prob = prob_dict[newstr]
                    if prob > 0.0:
                        print("[{}|{}] {:0.6f}".format(tags_,tags,prob)),
            print"\n"
    
'''
def transition_prob2(train_name, uq_tags):
    tags_ = []
    curr_count = defaultdict(int)
    with open(train_name) as f:
	for line in f:
	    line = line.strip().split()
	    if len(line)>1:
	        tags_.append(line[1])
	        curr_count[line[1]]+=1
	    else:
	        tags_.append(' ')
	        curr_count[' ']+=1
	
    tags_.append(' ')
    curr_count[' ']+=1
	
    curr_prev_count = defaultdict(int) 
	
    for prev_, curr_ in zip(tags_[1:],tags_[:-1]):
	curr_prev_count[(curr_, prev_)] += 1 

    unique_tags = set(curr_count.keys())
    unique_tags = sorted(unique_tags)
    tag_priors = defaultdict(float)
    transition_probs = defaultdict(float)

	
    for curr in unique_tags:
	total_prob = 0
	for prev in unique_tags:
	    curr_prob=(curr_prev_count[(curr, prev)]/curr_count[curr])
	    total_prob += curr_prob
	    transition_probs[(prev, curr)] = curr_prob
		
	if curr!='``':
	    print("[ {:0.6f} ]".format(total_prob)),
	    tag_priors[curr] = total_prob
	    for tgs in unique_tags:
	        if transition_probs[(tgs, curr)]>0.0:
	            print("[{}|{}] {:0.6f}".format(tgs,curr,transition_probs[(tgs, curr)])),
	    print("\n")	

		
    return tag_priors, transition_probs	  
'''    
def check_test_in_corpus(test,emission_probs):
    
    lines = test.strip().split()
    
    for line in lines:
        tags_list = [tag for words, tag in emission_prob if words==line]
        tags_list.sort()
        print "\t"+line.title()+":\t",
        for tags in tags_list:
            probs = emission_probs[(line,tags)]
            print str(tags)+"("+'{0:.6f}'.format(probs)+")",
         
        print "\n"

def viterbi_algo(test,emission_prob,transition_prob,uq_tokens,uq_unigram):
    words  = test.strip().split()
    max_words_tags = []
    prev_tags_dict = dict()
    for i, word in enumerate(words):
        tags_list = [tag for wrd, tag in emission_prob if wrd==word] 
        tags_list.sort()
        sum_probs = []
        tags_t = []
        tags_sum_probs = dict()
        
        if i == 0:
            print"\nIteration"+' '+str(i)+":\t"+str(word)+" :",
            for tags in tags_list:
                e_probs = emission_prob[(word,tags)]
                str_key = str('')+' '+str(tags)
                t_probs = transition_prob[str_key]
                sum_probs.append((e_probs*t_probs))
                tags_t.append(tags)
                
            for t, tg in enumerate(tags_t):
                tags_sum_probs[tg] = sum_probs[t]/(sum(sum_probs))
                print str(tags_t[t])+"("+'{0:.6f}'.format(sum_probs[t]/(sum(sum_probs)))+", null) ",
            max_index, max_value = max(enumerate(sum_probs), key=operator.itemgetter(1))
            max_words_tags.append(tags_t[max_index])
            prev_tags_dict = tags_sum_probs
        
            print"\n"
        else:
            print"\nIteration"+' '+str(i)+":\t"+str(word)+" :",

            for tags in tags_list:
    
                max_prev_tags = max_words_tags[-1]
                e_probs = emission_prob[(word,tags)]
                if tags in prev_tags_dict:
                    str_key = str(tags)+' '+str(max_prev_tags)
                    t_probs = transition_prob[str_key]
                    prev_prob = prev_tags_dict[tags]
                    sum_probs.append((e_probs*t_probs*prev_prob))
                    tags_t.append(tags)
        
                else:
                    str_key_init = str('')+' '+str(tags)
                    t_probs_init = transition_prob[str_key_init]
                    sum_probs.append((e_probs*t_probs_init))
                    tags_t.append(tags)
            for t, tg in enumerate(tags_t):
                tags_sum_probs[tg] = sum_probs[t]/(sum(sum_probs))
                print str(tags_t[t])+"("+'{0:.6f}'.format(sum_probs[t]/(sum(sum_probs)))+","+str(max_prev_tags)+") ",
            max_index, max_value = max(enumerate(sum_probs), key=operator.itemgetter(1))
            max_words_tags.append(tags_t[max_index])
            prev_tags_dict = tags_sum_probs
            print"\n"
                
                
    return max_words_tags
  

ft = open(args.test,'r')
test = ft.read()
ft.close()

test = test.lower()

tokens,tags_corpus = read_train(args.train)

print "\nAll Tags Observed:\n\n"
uq_tags = tags_observed(tags_corpus)


sentences,tags,data = sentences_tags_corps(args.train)


print"\nInitial Distribution:\n\n"
init_dist = initial_dist(uq_tags,tags)


print"\nEmission Probabilities:\n\n"
emission_prob = emission_prob(tokens,tags_corpus)

#tag_priors, transition_probs = transition_prob2(args.train,uq_tags)

unigram,bigrams,words = cal_bigrams_lexicals(data)

print"\nTransition Probabilities:\n\n"
#print transition probs
print_transition_prob(bigrams,sorted(unigram.keys()))



print "\nCorpus Features:\n"
print "\n   Total # tags\t\t:",len(uq_tags)
print "\n   Total # bigrams\t:",len(bigrams)
print "\n   Total # lexicals\t:",len(list(set(tokens)))
print "\n   Total # sentencess\t:",len(sentences)

print "\n\nTest Set Tokens Found in Corpus:\n"
check_test_in_corpus(test,emission_prob)


print"\nIntermediate Results of Viterbi Algorithm:\n"


max_tags = viterbi_algo(test,emission_prob,bigrams,list(set(tokens)),uq_tags)

print"\nViterbi Tagger Output:\n"

for w, wrd in enumerate(test.strip().split()):
    print"\t\t"+wrd+"\t"+str(max_tags[w])

