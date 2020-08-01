import json
from time import sleep

import requests

_description_query = '''
SELECT ?item ?itemDescription
WHERE 
{
  OPTIONAL {wd:%s wdt:P31 ?item.}
  OPTIONAL {wd:%s wdt:P279 ?item.}
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
'''


def get_instance_descriptions(word):
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    descriptions = []
    for query in [_description_query]:
        sleep(1)
        try:
            data = requests.get(url, params={'query': query % (word, word), 'format': 'json'}).json()
        except json.decoder.JSONDecodeError:
            return []
        for item in data['results']['bindings']:
            try:
                descriptions.append(item['itemDescription']['value'])
            except KeyError:
                pass

    return descriptions


def load_indices_dict(filename):
    lines = open(filename).readlines()
    indices_dict = {}
    for line in lines:
        pos = line.find(' [')
        key = line[:pos]
        value = eval(line[pos:].strip())
        indices_dict[key] = value

    return indices_dict