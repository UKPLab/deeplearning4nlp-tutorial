Training Data for SemEval-2 Task #8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals

Iris Hendrickx, Su Nam Kim, Zornitsa Kozareva, Preslav Nakov, Diarmuid Ó Séaghdha, Sebastian Padó, Marco Pennacchiotti, Lorenza Romano and Stan Szpakowicz

The accompanying dataset is released under a Creative Commons Atrribution 3.0 Unported Licence (http://creativecommons.org/licenses/by/3.0/).

Version 1.0: March 5, 2010


SUMMARY

This dataset consists of 8000 sentences which have been annotated according to the scheme for SemEval-2 Task #8. Some sentences have been reused from SemEval-1 Task #4 (Classification of Semantic Relations between Nominals), the rest have been collected from the Web specifically for this task. All have been annotated in accordance with the new relation definitions included in this data release.


RELATIONS

We chose nine relations for SemEval-2 Task #8:

(1) Cause-Effect
(2) Instrument-Agency
(3) Product-Producer
(4) Content-Container
(5) Entity-Origin
(6) Entity-Destination
(7) Component-Whole
(8) Member-Collection
(9) Message-Topic

Relations 1-5 also featured in SemEval-1 Task #4, and a subset of the positive and negative examples in the SemEval-1 Task #4 dataset for these relations are included in the training data here. The definitions for all nine relations appear in the files Task8_Relation*.pdf (also included in the distribution).


DATA FORMAT

The format of the data is illustrated by the following examples:

15 "They saw that the <e1>equipment</e1> was put inside rollout <e2>drawers</e2>, which looked aesthetically more pleasing and tidy."
Content-Container(e1,e2)
Comment: the drawer contains the equipment, typical example of Content-Container; no movement

20 ""Any time," he told her before turning to the <e1>boy</e1> who was in the <e2>desk</e2> next to him."
Other
Comment: the desk does not contain the boy.

The first line contains the sentence itself inside quotation marks, preceded by a numerical identifier. Each sentence is annotated with three pieces of information:

(a) Two entity mentions in the sentence are tagged as e1 and e2 -- the numbering simply reflects the order of the mentions in the sentence. The span of the tag corresponds to the "base NP" which may be smaller than the full NP denoting the entity.

(b) If one of the semantic relations 1-9 holds between e1 and e2, the sentence is labelled with this relation's name and the order in which the relation arguments are filled by e1 and e2. For example, Cause-Effect(e1,e2) means that e1 is the Cause and e2 is the Effect, whereas Cause-Effect(e2,e1) means that e2 is the Cause and e1 is the Effect. If none of the relations 1-9 holds, the sentence is labelled "Other". In total, then, 19 labels are possible.

(c) A comment may be provided to explain why the annotators chose a given label. Comments are intended for human readers and should be ignored by automatic systems participating in the task. Comments will not be released for the test data.

Note that the test release will be formatted similarly, but without lines for the relation label and for the comment.

Further information on the annotation methodology can be found in the enclosed document Task8_Guidelines.pdf.


EVALUATION

The task is to predict, given a sentence and two tagged entities, which of the relation labels to apply. Hence, the gold-standard labels (Cause-Effect(e1,e2) and so on) should be provided to a system at training time but not at test time. The predictions of the system should be in the following format:

1 Content-Container(e2,e1)
2 Other
3 Entity-Destination(e1,e2)
...

The official evaluation measures are accuracy over all examples and macro-averaged F-score over the 18 relation labels apart from Other. To calculate the F-score, 18 individual F-scores -- one for each relation label -- are calculated in the standard way and the average of these scores is taken. For each relation Rel, each sentence labelled Rel in the gold standard will count as either a true positive or a false negative, depending on whether it was correctly labelled by the system; each sentence labelled with a different relation or with Other will count as a true negative or false positive.


TEST PROCEDURE

The test data will be released on March 18. After this, participants will be able to download it at any time up to the final results submission deadline (April 2, 2010). Once the data have been downloaded, participants will have 7 days to submit their results; they must also submit by the final deadline of April 2. Late submissions will not be counted. Participants should supply four sets of predictions for the test data, using four subsets of the training data:

TD1   training examples 1-1000
TD2   training examples 1-2000
TD3   training examples 1-4000
TD4   training examples 1-8000

For each training set, participants may use the data in that set for any purpose they wish (training, development, cross-validation and so forth). However, the training examples outside that set (e.g., 1001-8000 for TD1) may not be used in any way. The final 891 examples in the training release (examples 7110-8000) are taken from the SemEval-1 Task #4 datasets for relations 1-5 and hence their label distribution is skewed towards those relation classes.  Participants have the option of including or excluding these examples as appropriate for their chosen learning method.

There is no restriction on the external resources that may be used.


USEFUL LINKS:

Google group: http://groups.google.com.sg/group/semeval-2010-multi-way-classification-of-semantic-relations?hl=en
Task website: http://docs.google.com/View?docid=dfvxd49s_36c28v9pmw
SemEval-2 website: http://semeval2.fbk.eu/semeval2.php


TASK SCHEDULE

 * Test data release:                        March 18, 2010
 * Result submission deadline:               7 days after downloading the *test* data, but no later than April 2
 * Organizers send the test results:         April 10, 2010
 * Submission of description papers:         April 17, 2010
 * Notification of acceptance:               May 6, 2010
 * SemEval-2 workshop (at ACL):              July 15-16, 2010
