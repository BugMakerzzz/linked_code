import csv,json,os,re,random
#这一系列载入数据都是载入平坦的三元组数据
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import create_index
import faiss

template = {
    "ObjectUse": "{0} is used for {1}.",

    "AtLocation": "{0} is located at {1}.",

    "MadeUpOf": "{0} is made up of {1}.",

    "HasProperty": "{0} has property {1}.",

    "CapableOf": "{0} can {1}.",
  
    "Desires": "{0} desires {1}.",

    "NotDesires": "{0} not desires {1}.",

    "isAfter": "{0} is after {1}.",
    
    "HasSubEvent": "{0} has sub event {1}.",

    "isBefore": "{0} is before {1}.",

    "HinderedBy": "{0} can be hindered by {1}.",

    "Causes": "{0} causes {1}.",

    "xReason": "{0} because {1}.",

    "isFilledBy": "{0}, blank can be filled by {1}.",  
    
    "xNeed": "{0}. PersonX needed {1}.",

    "xAttr": "{0}. PersonX has attribute of {1}.",

    "xEffect": "{0}. PersonX will {1}.",

    "xReact": "{0}. PersonX feels {1}.",

    "xWant": "{0}.PersonX wants {1}.", 

    "xIntent": "{0}. PersonX intents {1}.",

    "oEffect": "{0}. PersonY will {1}.",

    "oReact": "{0}. PersonY feels {1}.",

    "oWant": "{0}. PersonY wants {1}."

  }
  


def load_atomic2020_dataset():
    data_path = './data/atomic2020'
    splits = ['train','dev','test']
    data = []
    for split in splits:
        with open(os.path.join(data_path, f"{split}.tsv"), encoding='UTF-8') as f:
            source_reader = csv.reader(f,dialect='excel-tab')
            for line in source_reader:
                # line = line.rstrip('\n').split('\t')
                #can do some normalize here
                if line[0] and line[1] and line[2]:
                    h,r,t = line
                    #normalization
                    h = normalize_tail(h)
                    t = normalize_tail(t)
                    if t == "none": # 2022.3.4 Sept 除掉tail中的none
                        continue
                    if match_blank(h): # 2022.3.4 Sept 除掉所有有待填空的知识
                        continue
                    # h = change_person(h)
                    # t = change_person(t)
                    # pattern = 'PersonX'
                    # rex = re.search(pattern, h) 
                    # if not rex: # Sept 除掉所有不含PersonX的
                    #     continue
                    # val = h + " " + t
                    # if len(val.split(" ")) > 8: # Sept 除掉所有长度大于8的
                    #     continue
                    sentence = template[r].format(h,t)
                    d = [h, r, t, sentence] # 0:head, 1:relation, 2:tail, 3:label
                    data.append(d)
    return data
    

def normalize_tail(tail):
    _norm_symbols = {'\t':' ',
    '`':"'",
    '´':"'",
    'é':'e',
    'ê':'e',
    '—':'_',
    '’':"'"
    }
    _strip_symbols = ''.join({
    '\t',
    ' ',
    "'",
    ',',
    '.',
    '/',
    ':',
    '=',
    '>',
    '\\',
    '`',
    '´',
    '—',
    '’',
    })

    subline_regex = re.compile(r'_+')
    multispace_regex = re.compile(r' +')
    personx_regex = re.compile(r'\b[Pp]erson[Xx]\b|\b[Pp]erson [Xx]\b|\b[Xx]\b')
    persony_regex = re.compile(r'\b[Pp]erson[Yy]\b|\b[Pp]erson [Yy]\b|\b[Yy]\b')
    personz_regex = re.compile(r'\b[Pp]erson[Zz]\b|\b[Pp]erson [Zz]\b|\b[Zz]\b')
    tail = tail.strip(_strip_symbols)
    tail = ''.join([_norm_symbols[w] if w in _norm_symbols else w for w in tail])
    tail = subline_regex.sub('___',tail) #不做进一步处理了，原汁原味  2020.3.4 Sept：这步就是将所有横线都变成三个来统一格式，下一步同理
    tail = multispace_regex.sub(' ',tail)
    #don't do case mapping
    # if tail in case_mapping:
    #     tail = case_mapping[tail]
    tail = personx_regex.sub("PersonX",tail)
    tail = persony_regex.sub("PersonY",tail)
    tail = personz_regex.sub("PersonZ",tail)
    tail = 'none' if tail.lower() == 'none' else tail #合并所有none
    return tail

def add_mask(str):
    sent_list = str.split(" ")
    for i in range(len(sent_list)):
        if sent_list[i] == "___" or sent_list[i] == "__":
            sent_list[i] = "[MASK]"
    return (" ").join(sent_list)

def match_blank(str):
    pattern = re.compile(r'___')
    result = pattern.search(str)
    return result

def change_person(str):
    personx_regex = re.compile('personX', re.IGNORECASE)
    persony_regex = re.compile('personY', re.IGNORECASE)
    personz_regex = re.compile('personZ', re.IGNORECASE)
    str = re.sub(personx_regex, "Tom", str)
    str = re.sub(persony_regex, "Jerry", str)
    str = re.sub(personz_regex, "Jack", str)
    return str


def embedding_knowledge():
    result_path = './data/atomic2020/knowledge_sentence.json'
    index_path = './data/atomic2020/knowledge_sentence.index'
    model_path = './model/all-mpnet-base-v2'
    model = SentenceTransformer(model_path).cuda()
    data = load_atomic2020_dataset()
    results = []
    data_embeddings = []
    for tup in tqdm(data):
        sentence = tup[-1]
        message = {'knowledge':sentence}
        results.append(message)
        embedding = model.encode(sentence)
        data_embeddings.append(embedding)
    data_embeddings = np.array(data_embeddings)
    faiss_index = create_index(data_embeddings)
    faiss.write_index(faiss_index, index_path) 
    with open(result_path, 'w') as f:
        json.dump(results,f, indent=4)   
        f.close()

embedding_knowledge()