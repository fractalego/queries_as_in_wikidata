import json
import os
import random
import sys

import torch
from torch import nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from qfw.model import QueryModel

_path = os.path.dirname(__file__)
_train_filename = os.path.join(_path, '../data/wikidata-disambig-train.json')
_dev_filename = os.path.join(_path, '../data/wikidata-disambig-dev.json')
_description_filename = os.path.join(_path, '../data/indices.txt')
_rel_dict_filename = os.path.join(_path, '../data/relations_to_english.csv')
_save_filename = os.path.join(_path, '../data/save')

MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')


def train(train_model, batches, optimizer, criterion):
    total_loss = 0.
    for i, batch in tqdm(enumerate(batches), total=len(batches)):
        inputs, start_targets, end_targets, lengths = batch[0], batch[1], batch[2], batch[3]
        optimizer.zero_grad()
        start, end = train_model(inputs.cuda(), lengths)

        loss1 = criterion(start, start_targets.cuda().float())
        loss2 = criterion(end, end_targets.cuda().float())
        loss = loss1 + loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results


def create_one_hot_vector(index, lenght):
    vector = [0.] * lenght
    vector[index] = 1.
    return vector


def load_positive_data(filename, description_dict, example_id='correct_id', limit=-1):
    all_data = []
    disambig_data = json.load(open(filename))
    random.shuffle(disambig_data)
    for example in tqdm(disambig_data[:limit]):
        item_id = example[example_id]
        if item_id not in description_dict:
            continue
        query = random.choice(description_dict[item_id])
        sentence = example['text']
        entity = example['string']

        query_tokens = tokenizer.encode(query, add_special_tokens=False)
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        input_ids = torch.tensor([[101] + query_tokens
                                  + [102] + sentence_tokens
                                  + [102]
                                  ])
        entity_tokens = tokenizer.encode(entity, add_special_tokens=False)
        try:
            start, end = find_sub_list(entity_tokens, sentence_tokens)[0]
            end = end + 1
        except IndexError:
            continue

        length = len(sentence_tokens) + 2
        start_label = torch.tensor(create_one_hot_vector(start + 1, length))
        end_label = torch.tensor(create_one_hot_vector(end + 1, length))
        query_length = len(query_tokens) + 1

        all_data.append((input_ids, start_label, end_label, query_length))

    return all_data


def load_negative_data(filename, description_dict, example_id='wrong_id', limit=-1):
    all_data = []
    disambig_data = json.load(open(filename))
    random.shuffle(disambig_data)
    for example in tqdm(disambig_data[:limit]):
        item_id = example[example_id]
        if item_id not in description_dict:
            continue
        query = description_dict[item_id]
        sentence = example['text']

        query_tokens = tokenizer.encode(query, add_special_tokens=False)
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        input_ids = torch.tensor([[101] + query_tokens
                                  + [102] + sentence_tokens
                                  + [102]
                                  ])
        start, end = 0, 0
        length = len(sentence_tokens) + 2
        start_label = torch.tensor(create_one_hot_vector(start, length))
        end_label = torch.tensor(create_one_hot_vector(end, length))
        query_length = len(query_tokens) + 1

        all_data.append((input_ids, start_label, end_label, query_length))

    return all_data


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def batchify(data, n):
    len_dict = {}
    for item in data:
        in_length = item[0].shape[1]
        out_length = item[-1]
        try:
            len_dict[(in_length, out_length)].append(item)
        except:
            len_dict[(in_length, out_length)] = [item]

    batch_chunks = []
    for k in len_dict.keys():
        vectors = len_dict[k]
        batch_chunks += chunks(vectors, n)

    batches = []
    for chunk in batch_chunks:
        input = torch.stack([item[0][0] for item in chunk])
        labels1 = torch.stack([item[1] for item in chunk])
        labels2 = torch.stack([item[2] for item in chunk])
        labels3 = torch.tensor([item[-1] for item in chunk])
        batches.append((input, labels1, labels2, labels3))

    return batches


def load_indices_dict(filename):
    lines = open(filename).readlines()
    indices_dict = {}
    for line in lines:
        pos = line.find(' [')
        key = line[:pos]
        value = eval(line[pos:].strip())
        indices_dict[key] = value

    return indices_dict


def test(eval_model, batches, tokenizer, threshold=0.5):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i, batch in tqdm(enumerate(batches), total=len(batches)):
        inputs, start_targets, end_targets, lengths = batch[0], batch[1], batch[2], batch[3]
        starts, ends = eval_model(inputs.cuda(), lengths)

        for input, start_target, end_target, start_ohv, end_ohv, length \
                in zip(inputs, start_targets, end_targets, starts, ends, lengths):

            start_target = torch.argmax(start_target)
            end_target = torch.argmax(end_target)

            words = [tokenizer.convert_ids_to_tokens([i])[0] for i in list(input)]

            adversarial_score = min(start_ohv[0], end_ohv[0])
            model_says_it_has_answer = adversarial_score < threshold
            there_is_an_answer = start_target != 0


            if not there_is_an_answer and not model_says_it_has_answer:
                tn += 1
                continue

            if not there_is_an_answer and model_says_it_has_answer:
                fp += 1
                continue

            if there_is_an_answer and not model_says_it_has_answer:
                fn += 1
                continue

            if model_says_it_has_answer:
                start = torch.argmax(start_ohv[1:]) + 1
                end = torch.argmax(end_ohv[1:]) + 1
                if start == start_target and end == end_target:
                    tp += 1
                    #print('tp:', ' '.join(words))
                    #print(words[length + start:length + end])
                    #print(words[length + start_target:length + end_target])

    precision = 0
    if tp or fp:
        precision = tp / (tp + fp)

    recall = 0
    if tp or fn:
        recall = tp / (tp + fn)

    f1 = 0
    if precision or recall:
        f1 = 2 * precision * recall / (precision + recall)

    print('   precision:', precision)
    print('   recall:', recall)
    print('   F1:', f1)

    return f1


if __name__ == '__main__':
    model_class, tokenizer_class, pretrained_weights = MODEL
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    language_model = model_class.from_pretrained(pretrained_weights)

    print('Loading training data')
    _limit = 50000
    description_dict = load_indices_dict(_description_filename)
    train_data = load_positive_data(_train_filename, description_dict, limit=_limit) \
                 + load_negative_data(_train_filename, description_dict, limit=_limit)
    random.shuffle(train_data)
    train_batches = batchify(train_data, 10)

    print('Loading validation data')
    dev_data = load_positive_data(_dev_filename, description_dict, limit=_limit) \
               + load_negative_data(_dev_filename, description_dict, limit=_limit)
    random.shuffle(dev_data)
    dev_batches = batchify(dev_data, 10)

    train_model = QueryModel(language_model)
    train_model.cuda()

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-5)

    for epoch in range(20):
        random.shuffle(train_batches)
        train_model.train()
        loss = train(train_model, train_batches, optimizer, criterion)
        print('Epoch:', epoch, 'Loss:', loss)

        train_model.eval()
        test(train_model, dev_batches, tokenizer)

        torch.save({
            'epoch': epoch,
            'model_state_dict': train_model.state_dict()},
            _save_filename + str(epoch))

        sys.stdout.flush()
