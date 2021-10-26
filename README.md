# powerfunction-learner
 
To run the learner from the command line, enter python3 power_learn.py input.txt iter lr_weight lr_power

The input file should have the same format as MaxEnt Grammar Tool input files  (https://linguistics.ucla.edu/people/hayes/MaxentGrammarTool/). Please refer tp the manual p.2-3 (https://linguistics.ucla.edu/people/hayes/MaxentGrammarTool/ManualForMaxentGrammarTool.pdf).

There are parameters that can be specified optionally. 
Iter is the number of maximum iterations (default set to 30,000). 
lr_weight is the learning rate of the weight (default set to 0.1).
lr_power is the learning rate of the exponents (default set to 0.01). 