import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.get_logger(__name__)
import os

MAX_LEN = 35000000 # it is only used for logs, not important if it is not precise

CHUNK = 100000
conceptnet_path = 'conceptnet/data/conceptnet-assertions-5.7.0.csv'
output_file = 'conceptnet/data/conceptnet-en.csv'

df_iter = pd.read_csv(conceptnet_path, sep='\t', header=None, 
                      names=["assertion", "predicate", "subject", "object", "metadata"], 
                      index_col = None, iterator=True, chunksize=CHUNK)


open(output_file, 'w+').close()

total = 0
last = False
while not last:
    logger.info(f'Processed {int((total/MAX_LEN)*100)}%')
    df = df_iter.get_chunk()
    total += CHUNK
    logger.info(f'Got {len(df)}')
    if len(df) < CHUNK:
        last = True
    
    df = df[df.apply(lambda r: ('/c/en/' in r['subject']) and ('/c/en/' in r['object']), axis=1)]
    if not len(df):
        continue
    
    df[['subject', 'object', 'predicate', 'metadata']].to_csv(output_file, mode='a', header=False, index=False)
