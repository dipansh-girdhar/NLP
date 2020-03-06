Methodology:


The same data is used as the input for the creation of spacy based custom NER.

The difference in the case of spacy is that spacy doesn’t accept the general .conll format NER data as input. So, first, the data is converted into spacy accepted JSON (dictionary) format.

The normal .conll NER format:         
![NER_conll_format](NER/Spacy NER/NER_conll.png?raw=true "NER_conll_format")

Spacy accepted NER format:
![spacy_accepted](NER/Spacy NER/spacy_accepted_ner.png?raw=true "spacy_accepted")


The sentences are created based on the appearance of “. 0” in the .conll file which marked the end of a sentence.

The spacy NER is created on a blank en model using the data converted into the required format.

While training the spacy ner, all the other functionalities in the spacy pipeline are disabled.

The spacy trained NER is tested on the same test data using the Scorer method from the spacy library.
