import os
import corenlp
import os
import itertools

def coreference(sentence):
    #pass in max 5 sentences!
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.coref

    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
    x = predictor.predict(
      document=sentence
    )
    document = x['document']
    clusters = x['clusters']
    if clusters != []:
        print(clusters)
        for cluster in clusters:
            # [[0, 1], [8, 8]
            for index in range(cluster.__len__() - 1):
                replace = ' '.join(document[cluster[0][0]:cluster[0][1] + 1])
                print(replace)
                start = cluster[index + 1][0]
                end = cluster[index + 1][1]
                print(document[start:end + 1])
                del document[start:end + 1]
                document.insert(start, replace)
    string = ' '.join(document)
    return string

os.environ["CORENLP_HOME"] = '/Users/dj_jnr_99/Downloads/stanford-corenlp-full-2018-10-05'

def test_tokensregex(text):
    with corenlp.CoreNLPClient(annotators='tokenize ssplit ner depparse'.split(), timeout=60000) as client:
        # Example pattern from: https://nlp.stanford.edu/software/tokensregex.shtml
        pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
        matches = client.tokensregex(text, pattern)
        print(matches)


def filter_triples(triples): #NOT SURE IF THIS GETS RIGHT RESULT YET

    subjects = []
    relations = []
    objects = []

    for triple in triples:

        subjects.append(triple['subject'])
        relations.append(triple['relation'])
        objects.append(triple['object'])

    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            if subjects[i] in subjects[j]:
                subjects[i] = subjects[j]
            elif subjects[j] in subjects[i]:
                subjects[j] = subjects[i]

    for i in range(len(relations)):
        for j in range(i + 1, len(relations)):
            if relations[i] in relations[j]:
                relations[i] = relations[j]
            elif relations[j] in relations[i]:
                relations[j] = relations[i]

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if objects[i] in objects[j]:
                objects[i] = objects[j]
            elif objects[j] in objects[i]:
                objects[j] = objects[i]
    newtriples = []
    for x in range(subjects.__len__()):
        newtriples.append({'subject': subjects[x], 'relation': relations[x], 'object': objects[x]})

    newtriples = [dict(s) for s in set(frozenset(d.items()) for d in newtriples)]
    for x in range(newtriples.__len__()):
        if newtriples[x]['object'] in newtriples[x]['relation']: #['Reduced Aβ load', 'shifting precursor protein processing toward', 'precursor protein processing'] -> ['Reduced Aβ load', 'shifting toward', 'precursor protein processing']
            #print(newtriples[x]['relation'], newtriples[x]['object'])
            newtriples[x]['relation'] = newtriples[x]['relation'].replace(newtriples[x]['object'],'')
    return newtriples

def text_annotate(text):
    with corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
        ann = client.annotate(text, properties={
        'annotators': 'tokenize, ssplit, pos, depparse, parse, openie',
        'outputFormat': 'json'
        })
        core_nlp_output = client.annotate(text=text, annotators=['openie'], output_format='json')
        triples = []

        for sentence in core_nlp_output['sentences']:
            for triple in sentence['openie']:
                triples.append({
                    'subject': triple['subject'],
                    'relation': triple['relation'],
                    'object': triple['object']
                })

        return triples

text= 'Donepezil has been reported to have limited Aβ-targeting mechanisms beside its acetylcholine esterase inhibition. The aim of this study was to investigate the consumption of EVOO rich with oleocanthal (hereafter EVOO) as a medical food on enhancing the effect of donepezil on attenuating Aβ load and related toxicity in 5xFAD mouse model of AD. Our results showed that EVOO consumption in combination with donepezil significantly reduced Aβ load and related pathological changes. Reduced Aβ load could be explained, at least in part, by enhancing Aβ clearance pathways including blood-brain barrier (BBB) clearance and enzymatic degradation, and shifting amyloid precursor protein processing toward the nonamyloidogenic pathway. Furthermore, EVOO combination with donepezil up-regulated synaptic proteins, enhanced BBB tightness and reduced neuroinflammation associated with Aβ pathology. In conclusion, EVOO consumption as a medical food combined with donepezil offers an effective therapeutic approach by enhancing the noncholinergic mechanisms of donepezil and by providing additional mechanisms to attenuate Aβ-related pathology in AD patients.'



from nltk import sent_tokenize
sents = sent_tokenize(text)

for sent in sents:
    triples = text_annotate(sent)
    triples = filter_triples(triples)
    for triple in triples:
        print([triple['subject'],triple['relation'],triple['object'], sent])
    print("")

def get_relations(text, filename):

    allrelations = []
    sents = sent_tokenize(text) #tokenize sentences

    for sent in sents:
        triples = text_annotate(sent) #retrieve triples
        triples = filter_triples(triples) #delete unnecessary triple duplicates

        for triple in triples:
            allrelations.append([triple['subject'], triple['relation'], triple['object'], sent, filename])

    return allrelations
