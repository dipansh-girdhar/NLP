
#pip install flair

from flair.datasets import ColumnCorpus
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, WordEmbeddings, CharacterEmbeddings, FlairEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

'''
Running a sample:
"python train_flair.py --input_dir=/opt/ml/Input --model_dir=/opt/ml/model --train_file=ner_data_train.conll --test_file=ner_data_test.conll --dev_file=ner_data_dev.conll"
'''
# input_dir = '/opt/ml/input'
# model_dir = '/opt/ml/model'
# output_dir = '/opt/ml/output'

def train(args, tag_type):
    '''
    Training script to be run for training the ner model

    Parameters:
    -----------
    args:arguments passed to the parser on CLI
    '''
    data_dir = args.input_dir + '/data'
    corpus =ColumnCorpus(data_folder=data_dir,
                            column_format={0: 'text', 1: 'ner'},
                            train_file=args.train_file,
                            test_file=args.test_file,
                            dev_file=args.dev_file)

    # print(corpus.train[0])
    # print(corpus)

    # tag_type = 'ner'

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    # print(tag_dictionary)
    
    if args.character_embeddings:
        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('glove'),
            CharacterEmbeddings(),
            FlairEmbeddings(args.flair_model_name_or_path_forward),
            FlairEmbeddings(args.flair_model_name_or_path_backward),
        ]
    else:
        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('glove'),
            FlairEmbeddings(args.flair_model_name_or_path_forward),
            FlairEmbeddings(args.flair_model_name_or_path_backward),
        ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # initialize sequence tagger
    
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # start training
    trainer.train(args.model_dir,
                learning_rate=args.train_learning_rate,
                mini_batch_size=args.per_gpu_batch_size,
                max_epochs=args.num_train_epochs,
                embeddings_storage_mode=args.embeddings_storage_mode)

    model = SequenceTagger.load(args.model_dir + '/final-model.pt')
    if(args.predict_file):
        with open(data_dir + args.predict_file, 'r') as f:
            str_file=f.read()
        
        sentence = Sentence(str_file)
        
        model.predict(sentence)
        print(sentence.to_tagged_string())


def predict(args):
    '''
    Predict the named entities loading an existing model

    Parameters:
    -----------
    args:arguments passed to the parser on CLI
    '''
    # load the model
    model = SequenceTagger.load(args.model_dir + '/final-model.pt')
    # create example sentence
    if(args.predict_file):
        data_dir = args.input_dir + '/data/'
        with open(data_dir + args.predict_file, 'r') as f:
            str_file=f.read()
        sentence = Sentence(str_file)
    else:
        sentence=Sentence("This is a dummy sentence. To get results on your file and not just on this sentence, add predict_file argument while running")
    # predict tags and print
    model.predict(sentence)

    print(sentence.to_tagged_string())


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=None, type=str, help="Path to input directory")
    parser.add_argument("--model_dir", default=None, type=str, help="Path to model directory")
    parser.add_argument("--train_file", default=None, type=str, help="Train file")
    parser.add_argument("--test_file", default=None, type=str, help="Test file")
    parser.add_argument("--dev_file", default=None, type=str, help="Dev file")
    parser.add_argument("--flair_model_name_or_path_forward", default="news-forward", type=str, help='Flair forward pretrrained embeddings')
    parser.add_argument("--flair_model_name_or_path_backward", default="news-backward", type=str, help='Flair backward pretrrained embeddings')
    parser.add_argument("--character_embeddings", default=True, type=bool, help='Using character embeddings or not')
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--per_gpu_batch_size", default=32, type=int, help="batch size for training")
    parser.add_argument("--train_learning_rate", default=1e-2, type=float, help="Learning rate for training")
    parser.add_argument("--embeddings_storage_mode", default="gpu", type=str, help="Embeddings storage mode")
    parser.add_argument("--train_or_predict", default='train', type=str, help="Whether to train or predict. Either use 'train' or 'predict' as argument")
    parser.add_argument("--predict_file", default=None, type=str, help='Predict file, can be mentioned while training too')
    args=parser.parse_args()

    if(args.train_or_predict=='train'):
        train(args, tag_type='ner')
    elif(args.train_or_predict=='predict'):
        predict(args)



