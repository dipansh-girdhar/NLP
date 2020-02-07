from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.instance import Instance
from overrides import overrides

import itertools
from typing import Dict, List, Iterator

@DatasetReader.register("conll_03_reader")
class CONLL03Dataset Reader(DatasetReader):
    def __init__(self,
                token_indexers:Dict[str,TokenIndexer]= None,
                lazy:bool = False -> None:
        super().__init__(lazy))
        self.token_indexers=token_indexers or {'tokens': SingleIdTokenIndexer()}


    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as conll_file:
            # itertools.groupby is a powerful function that can group
            # successive items in a list by the returned function call.
            # In this case, we're calling it with `is_divider`, which returns
            # True if it's a blank line and False otherwise.
            for divider, lines in itertools.groupby(conll_file, is_divider):
                # skip over any dividing lines
                if divider: continue
                # get the CoNLL fields, each token is a list of fields
                fields = [l.strip().split() for l in lines]
                # switch it so that each field is a list of tokens/labels
                fields = [l for l in zip(*fields)]
                # only keep the tokens and NER labels
                tokens, _, _, ner_tags = fields

                yield self.text_to_instance(tokens, ner_tags)
