import argparse
from functools import partial
import hashlib
import json
import logging
from pathlib import Path
import pickle
from typing import Callable, List, Tuple
import warnings

import faiss
import numpy as np
import spacy
from tqdm import tqdm

from groundcskg.graphify.text_to_uri import english_filter, replace_numbers


CACHE_DIR = Path('.cache/')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
SPACY_MODEL='en_core_web_lg'


def init_cache():
    if not CACHE_DIR.exists():
        logger.debug(f'Creating cache dir at: {CACHE_DIR}')
        CACHE_DIR.mkdir(parents=True)


def _cache_path(fn, args, kwargs):
    fn_name = fn.__name__
    args_string = ','.join(str(arg) for arg in args)
    kwargs_string = json.dumps(kwargs)
    byte_string = fn_name + args_string + kwargs_string
    hash_object = hashlib.sha1(byte_string.encode())
    return CACHE_DIR / hash_object.hexdigest()


def cache():
    def decorator(fn):
        def load_cached_if_available(*args, **kwargs):
            path = _cache_path(fn, args, kwargs)
            if path.exists():
                logger.debug(f'Loading `{fn.__name__}` output from cache')
                with open(path, 'rb') as f:
                    return pickle.load(f)
            output = fn(*args, **kwargs)
            with open(path, 'wb') as f:
                pickle.dump(output, f, protocol=4)
            return output
        return load_cached_if_available
    return decorator


class Vocab:
    def __init__(self, words) -> None:
        self.idx_to_word = words
        self.word_to_idx = {word: idx for idx, word in enumerate(words)}


@cache()
def read_embedding_file(embedding_file: Path) -> Tuple[Vocab, np.ndarray]:

    logger.debug(f'Reading embeddings from {embedding_file}')

    with open(embedding_file, 'r') as f:
        info = next(f)
        shape = tuple(int(x) for x in info.split())
        embeddings = np.zeros(shape, dtype=np.float32)

        words = []
        for i, line in tqdm(enumerate(f), total=shape[0]):
            word, *embedding = line.split()
            embedding = np.array([float(x) for x in embedding])
            words.append(word)
            embeddings[i] = embedding

    vocab = Vocab(words)

    return vocab, embeddings


def build_index(metric: str, embeddings: np.ndarray):

    logger.debug(f'Building search index')

    if metric == 'cosine':
        index = faiss.IndexFlatIP(embeddings.shape[-1])
    elif metric == 'l2':
        index = faiss.IndexFlatL2(embeddings.shape[-1])
    else:
        raise ValueError(f'Bad metric: {metric}')

    index.add(embeddings)

    return index


def generate_instances(dataset: Path):
    with open(dataset, 'r') as f:
        for line in f:
            yield(json.loads(line))


def get_extraction_fn(extraction_strategy: str,
                      ngram_length: int) -> Callable[[List[str], Vocab], List[str]]:
    if extraction_strategy == 'exhaustive':
        return partial(exhaustive_extraction, ngram_length=ngram_length)
    elif extraction_strategy == 'greedy':
        return partial(greedy_extraction, ngram_length=ngram_length)
    elif extraction_strategy == 'root':
        return partial(root_extraction, ngram_length=ngram_length)
    else:
        raise ValueError(f'Bad extraction strategy: {extraction_strategy}')


def exhaustive_extraction(phrase: List[str],
                          vocab: Vocab,
                          ngram_length: int) -> List[str]:
    tokens = english_filter([x.lower() for x in phrase])
    num_tokens = len(tokens)
    out = []
    for n in range(1, ngram_length):
        for i in range(num_tokens - n + 1):
            concept = replace_numbers('_'.join(tokens[i: i+n]))
            if concept in vocab.word_to_idx:
                out.append(concept)
    return out


def greedy_extraction(phrase: List[str],
                      vocab: Vocab,
                      ngram_length: int) -> List[str]:
    tokens = english_filter([x.lower() for x in phrase])
    out = []
    while len(tokens) > 0:
        for n in range(ngram_length + 1, 0, -1):
            concept = replace_numbers('_'.join(tokens[:n]))
            if concept in vocab.word_to_idx:
                out.append(concept)
                tokens = tokens[n:]
                break
            elif n == 1:
                tokens = tokens[n:]
    return out


def root_extraction(phrase: List[str],
                    vocab: Vocab,
                    ngram_length: int) -> List[str]:
    doc = nlp(' '.join(phrase))

    # Logic for noun phrases
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ == 'ROOT':

            # Return entire chunk if in vocab
            tokens = english_filter([token.text.lower() for token in chunk])
            concept = replace_numbers('_'.join(tokens))
            if concept in vocab.word_to_idx:
                return [concept]

            # Otherwise return the just the root token
            root = replace_numbers(chunk.root.text.lower())
            if root in vocab.word_to_idx:
                return [root]

    # Logic for other types of roots
    for token in doc:
        if token.dep_ == 'ROOT':
            concept = replace_numbers(token.text.lower())
            if concept in vocab.word_to_idx:
                return [concept]

    return []


def link(graphs: List,
         output: Path = None,
         embedding_file: Path = Path('../numberbatch-en-19.08.txt'),
         emb_model: str = 'bert',
         metric: str = 'cosine',
         extraction_strategy: str = 'greedy',
         ngram_length: int = 3,
         num_candidates: int = 5,
         only_link_atoms: bool = False,
         debug: bool = False) -> List:
    """
    Browse the top-k conceptnet candidates for a node.

    Parameters
    ==========
    input : List
        A list containing parsed alpha NLI graphs.
    output : Path
        Jsonl file to serialize output to.
    embedding_file : Path
        A txt file containing the embeddings.
    emb_model: str
        A string indicating which model to encode the text with, one of 'ft' or 'bert'.
    metric: str
        Similarity metric. One of: 'cosine', 'l2'
    extraction_strategy: str
        Approach for extracting concepts from mentions. One of: 'exhaustive', 'greedy'
    ngram_length: int
        Max length of n-grams to consider during concept extraction.
    num_candidates : int
        Number of candidates to return.
    only_link_atoms : bool
        Will only link the atoms (e.g., word tokens, word pieces) in the graph.
    """
    assert metric in {'cosine', 'l2'}
    assert extraction_strategy in {'exhaustive', 'greedy', 'root'}

    if debug:
        logger.setLevel(logging.DEBUG)

    global nlp
    nlp = spacy.load(SPACY_MODEL)


    global txt_embedding
    if emb_model=='ft':
        import fasttext
        import fasttext.util
        fasttext.util.download_model('en', if_exists='ignore')
        ft = fasttext.load_model('cc.en.300.bin')
    elif emb_model=='bert':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('bert-large-nli-cls-token')

    init_cache()
    vocab, embeddings = read_embedding_file(embedding_file)
    faiss.normalize_L2(embeddings) 
    index = build_index(metric, embeddings)
    extraction_fn = get_extraction_fn(extraction_strategy, ngram_length)


    if output:
        output_file = open(output, 'w')
    output_instances=[]
    for instance in graphs:
        output_instance = instance.copy()
        if only_link_atoms:
            nodes = [x for x in instance['nodes'] if x['is_atom']]
        else:
            nodes = instance['nodes']
        for uri, node in nodes.items():
            # Extract concept tokens from phrase
            phrase = node['phrase']
            # concepts = extraction_fn(phrase, vocab)
            # concept_ids = np.array([vocab.word_to_idx[concept] for concept in concepts])

            if len(phrase) > 0:
                # if len(concept_ids) > 1:
                #     mean_embedding = np.mean(embeddings[concept_ids], axis=0, keepdims=True)
                #     query = np.concatenate((embeddings[concept_ids], mean_embedding), axis=0)
                #     # query = mean_embedding
                # else:
                #     query = embeddings[concept_ids]
                if emb_model=='ft':
                    embeddings = np.stack([ft[x] for x in phrase], axis=0)
                    query = np.mean(embeddings, axis=0, keepdims=True)
                elif emb_model=='bert':
                    print(phrase)
                    query=np.array([model.encode(' '.join(phrase))])
                    faiss.normalize_L2(query)
                scores, candidate_ids = index.search(query, num_candidates)

                # Convert from cosine similarity to distance
                if metric == 'cosine':
                    scores = 1 - scores

                # If phrase contains k concepts then the search returns k * num_candidates results.
                # Reduces to top-k
                scores = scores.flatten()
                candidate_ids = candidate_ids.flatten()
                top_k_indices = np.argsort(scores)[:num_candidates]
                scores = scores[top_k_indices]
                candidate_ids = candidate_ids[top_k_indices]
                output_instance['nodes'][uri]['candidates'] = []
                for candidate_id, score in zip(np.nditer(candidate_ids), np.nditer(scores)):
                    candidate = vocab.idx_to_word[candidate_id]
                    # if '#' in candidate:
                    #     logger.warning(f'Encountered candidate URI: "{candidate}". Due to preprocessing steps '
					# 					'used to produce the ConceptNet Numberbatch embeddings this is '
					# 					'likely a bad link, and will be skipped.')
                    #     continue
                    output_instance['nodes'][uri]['candidates'].append({
						'uri': candidate,
						'score': score.item()
                    })
            else:
                output_instance['nodes'][uri]['candidates'] = []
        output_instances.append(output_instance)
        if output:
            output_file.write(json.dumps(output_instance) + '\n')
    if output:
        output_file.close()
    return output_instances


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='JSON-Lines file containing the text graphs.')
    parser.add_argument('--output', type=Path, required=False, help='Output JSON-Lines file.')
    parser.add_argument('--embedding_file', type=Path, required=False,
                        help='KG node embeddings. Currently only ConceptNet Numberbatch supported.')
    parser.add_argument('--metric', type=str, default='cosine',
                        help='Distance metric used for nearest neighbor search. One of: "cosine", "l2".')
    parser.add_argument('--extraction_strategy', type=str, default='greedy',
                        help='Approach for extracting concepts from text. One of: "exhaustive", "greedy", "root".'
                             'Exhaustive extraction produces all viable n-grams, e.g., "high school" -> {"high", "school", "high_school"}. '
                             'Greedy search produces only the largest viable n-grams, e.g., "high school" -> {"high_school"}. '
                             'Root extraction only produces the root of the dependency parse, e.g., "his car" -> {"car"}.')
    parser.add_argument('--ngram_length', type=int, default=3,
                        help='Maximum length of ngrams to consider when converting text to KG nodes (i.e. ConceptNet concepts).')
    parser.add_argument('--num_candidates', type=int, default=5, help='Number of candidates to return.')
    parser.add_argument('--debug', action='store_true', help='Enables debug statements.')
    args = parser.parse_args()

    link(graphs=generate_instances(args.input),
         output=args.output,
         embedding_file=args.embedding_file,
         emb_model='bert',
         metric=args.metric,
         extraction_strategy=args.extraction_strategy,
         ngram_length=args.ngram_length,
         num_candidates=args.num_candidates,
         debug=args.debug)


if __name__ == '__main__':
    main()
