import os

import torch
from transformers import BertModel, BertTokenizer

from qfw.model import QueryModel

_path = os.path.dirname(__file__)
_pre_trained_filename = os.path.join(_path, '../data/save1')

MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')


def predict(eval_model, tokenizer, query, sentence, threshold=0.5):
    query_tokens = tokenizer.encode(query, add_special_tokens=False)
    sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
    input_ids = torch.tensor([[101] + query_tokens
                              + [102] + sentence_tokens
                              + [102]
                              ])
    length = len(query_tokens) + 1
    lengths = torch.tensor([length])
    starts, ends = eval_model(input_ids, lengths)
    start_ohv, end_ohv = starts[0], ends[0]
    adversarial_score = min(start_ohv[0], end_ohv[0])

    model_says_it_has_answer = adversarial_score < threshold
    if model_says_it_has_answer:
        start = torch.argmax(start_ohv[1:]) + 1
        end = torch.argmax(end_ohv[1:]) + 1
        return tokenizer.decode(input_ids[0][length + start:length + end]), 1 - float(adversarial_score)

    return '', float(adversarial_score)


if __name__ == '__main__':
    model_class, tokenizer_class, pretrained_weights = MODEL
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    language_model = model_class.from_pretrained(pretrained_weights)

    model = QueryModel(language_model)
    checkpoint = torch.load(_pre_trained_filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    query = 'phenomenon that needs controlling'
    sentence = 'The government may have to consider closing pubs in England to control the transmission of the coronavirus and to enable schools to reopen after the summer holidays, one of its top scientific advisers has said.'

    print(predict(eval_model=model,
                  tokenizer=tokenizer,
                  query=query,
                  sentence=sentence,
                  ))
