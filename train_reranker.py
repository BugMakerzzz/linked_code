import torch
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

TRAIN_DATA_PATH = './data/reward_train/siqa_train_data.json'
# DATA_PATH = './data/classifier_train/wino_rerank_data_4k.json'
# OUTPUT_DIR = "./result/deberta_large_classifier/question_classifier"
OUTPUT_DIR = "./result/reward_model/siqa"
MODEL_PATH = "/mnt/publiccache/huggingface/deberta-v3-large"
# MODEL_PATH = './model/reward-model-deberta-v3-large-v2'


# MICRO_BATCH_SIZE = 1 # this could actually be 5 but i like powers of 2
train_batch_size = 16
dev_batch_size = 16
# GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
# gradient_accumulation_steps = 4
epochs = 5  # we don't always need 3 tbh
learning_rate = 1e-5  # the Karpathy constant  # 256 accounts for about 96% of the data
val_ratio = 0.1 
warm_up_steps = 50

seed = 17
torch.manual_seed(seed)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
device = 'cuda:3'

# word_embedding_model = models.Transformer(MODEL_PATH)

# # Apply mean pooling to get one fixed sized sentence vector
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
#                                pooling_mode_mean_tokens=True,
#                                pooling_mode_cls_token=False,
#                                pooling_mode_max_tokens=False)

# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


model = CrossEncoder(MODEL_PATH, num_labels=1, max_length=512)



with open(TRAIN_DATA_PATH, 'r') as f:
    js_data = json.load(f)
    f.close()

val_set_size = int(len(js_data) * val_ratio)


dev_data = [] 
i = 0

while i < val_set_size:
    tup = js_data[i]
    question = tup['question']
    j = 0
    message = None
    while i+j < val_set_size and  js_data[i+j]['question'] == question:
        if j == 0:
            message = {'query': question, 'positive': set(), 'negative': set()}
        tup = js_data[i+j]
        label = tup['label']
        if label == 0:
            message['positive'].add(tup['knowledge'])
        else:
            message['negative'].add(tup['knowledge'])
        j += 1
    if message:
        dev_data.append(message)
    i += j

train_data = []
for i in range(val_set_size,len(js_data)):
    tup = js_data[i]
    label = tup['label']
    if label == 0:
        label = 1
    elif label == 1:
        label = 0
    example = InputExample(texts=[tup['question'], tup['knowledge']], label=label)
    train_data.append(example)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
# train_loss = losses.CosineSimilarityLoss(model=model)
# evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_data)
evaluator = CERerankingEvaluator(dev_data, name='train-eval')

model = model

model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=epochs,
          evaluation_steps=50,
          warmup_steps=warm_up_steps,
          output_path=OUTPUT_DIR,)

