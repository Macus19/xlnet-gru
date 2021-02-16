path_to_jar = '../stanford-parser-full-2018-02-27/stanford-parser.jar'
path_to_models_jar = '../stanford-parser-full-2018-02-27/stanford-chinese-corenlp-2018-02-27-models.jar'

import json
import jieba
from ntlk.parse.stanford import StanfordParser, StanfordDependencyParser

nlist=['NN','NR','NT','CD','M']
vlist=['VV','VP','MD','COP']

# 数字转为向量
def getnum(x):
    x = int(x)
    num = [0] * 8
    pos = 0
    while x > 0:
        num[pos] = x % 10
        x =int (x/10)
        pos += 1
        if (pos > 7):
            break
    if (x > 9):
        num = [9] * 8
    return num

def get_dependency():
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar, model_path="")
    with open("/dataset/CAIL2018/data_train.json") as f:
        for line in f:
            json_data = json.loads(line)
            fact = jieba.cut(json_data["fact"], cut_all=False)
            result = dependency_parser.raw_parse(fact)
            dep = result.__next__()
            for triple in dep.triples():
                if (triple[0][1] in nlist and triple[2][1] in vlist) or (triple[0][1] in vlist and triple[2][1] in nlist) or (triple[0][1] in nlist and triple[2][1] in nlist):
		            print(triple)


if __name__ == "__main__":
    print(getnum(3000))
