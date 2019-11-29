# LSH-based-FAQ-Generation
Generates FAQ from a given Q/A dataset(Tested on amazon)
  
## Prerequisites:  
1.Python 2.7.10  
2.Numpy   
3.Scripy  
4.Pandas  
5.gzip  
6.glob  
7.nltk     
8.dateutil  
9.re  
10.pickle  


## How to run:
1. Clone/download the repository  
2. Ensure that Q/A dataset is from [here](http://jmcauley.ucsd.edu/data/amazon/qa/) if you want to test on a different dataset.
3. Execute python main.py

## Possible Error Scenario: ValueError: unknown locale: UTF-8
Add these commands to your ~/.bash_profile file:    
export LC_ALL="en_US.UTF-8"    
export LANG="en_US.UTF-8"  
#### Now in termeninal execute:  
source ~/.bash_profile

## References:
##### 1. For LSH [this](https://github.com/brandonrobertz/SparseLSH) code has been modified.
##### 2. Modeling ambiguity, subjectivity, and diverging viewpoints in opinion question answering systems  
##### Mengting Wan, Julian McAuley  
##### International Conference on Data Mining (ICDM), 2016  
##### 3. Addressing complex and subjective product-related queries with customer reviews  
##### Julian McAuley, Alex Yang  
##### World Wide Web (WWW), 2016  

