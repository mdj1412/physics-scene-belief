"""Audit public repository metadata and the source CSV overlap using Python and curl."""
import csv
import json
import subprocess
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMIT = '159c6f040dd7832570b537b3b5fd00a97fc28c15'
TARGET = ROOT.parents[1] / 'racket_transfer_v70/data/upstream'

def fetch(url):
    raw = subprocess.check_output(['curl','--fail','--silent','--show-error','--max-time','20',
                                   '--user-agent','v73-research-archive-index/1.0',url])
    return json.loads(raw)

def read_states(part):
    with (TARGET / part / 'racket_10.csv').open() as source:
        table = csv.reader(source)
        columns = next(table)
        return columns, Counter(tuple(map(float, row)) for row in table)

def main():
    result = {'utc': datetime.now(timezone.utc).isoformat()}
    url = f'https://api.github.com/repos/gossardt/tt_rackets/git/trees/{COMMIT}?recursive=1'
    tree = fetch(url)
    files = [row['path'] for row in tree['tree'] if row['type'] == 'blob']
    assert not tree['truncated']
    result['gossard'] = {'url': url, 'commit': tree['sha'], 'files': files, 'n_files': len(files)}
    columns, original = read_states('original')
    _, train = read_states('train')
    _, test = read_states('test')
    result['configuration_10'] = {'columns': columns, 'original':sum(original.values()),
        'train':sum(train.values()), 'test':sum(test.values()), 'original_not_in_splits':sum((original-train-test).values()),
        'split_rows_absent_from_original':sum((train+test-original).values())}
    result['datacite'] = []
    for query in ['titles.title:"table tennis"', '"table tennis" AND ("anti-spin" OR "antispin")',
                  '"table tennis" AND (racket OR rubber) AND (spin OR impact)']:
        address = 'https://api.datacite.org/dois?' + urllib.parse.urlencode({'query':query,
            'resource-type-id':'dataset','page[size]':250})
        record = {'query':query,'url':address}
        try:
            reply = fetch(address)
            record['total'] = reply['meta']['total']
            record['results'] = [{'doi': item['id'], 'title': item['attributes']['titles'][0]['title']}
                                 for item in reply['data']]
        except (OSError, subprocess.CalledProcessError) as error:
            record['error'] = str(error)
        result['datacite'].append(record)
    (ROOT/'archive_index.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':
    main()
