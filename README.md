# My_TS_working_framework

# Step 0: Install Datasets for Benchmark

1. https://github.com/SalesforceAIResearch/gift-eval  
2. https://github.com/SalesforceAIResearch/uni2ts  
3. https://github.com/decisionintelligence/TFB  
4. https://github.com/GestaltCogTeam/BasicTS  
5. https://github.com/TongjiFinLab/FinTSB  
6. https://github.com/microsoft/ProbTS
7. https://github.com/amazon-science/chronos-forecasting
8. https://github.com/PriorLabs/tabpfn-time-series
9. https://github.com/ibm-granite/granite-tsfm
   
or from my forks:

1. https://github.com/ChiShengChen/TFB
2. https://github.com/ChiShengChen/BasicTS
3. https://github.com/ChiShengChen/FinTSB
4. https://github.com/ChiShengChen/ProbTS
5. https://github.com/ChiShengChen/gift-eval
6. https://github.com/ChiShengChen/uni2ts
7. https://github.com/ChiShengChen/chronos-forecasting
8. https://github.com/ChiShengChen/tabpfn-time-series
9. https://github.com/ChiShengChen/granite-tsfm

`git clone https://github.com/SalesforceAIResearch/gift-eval.git`  
`cd gift-eval`  
`pip install -e .`  
`pip install -e .[baseline]`  
`pip install uni2ts`  `pip install gluonts==0.15.1`  
`pip install chronos-forecasting`  

Install timesfm package in a python 3.10.x env:
``
pip install timesfm[pax]
``
You can also try the torch version in a python 3.11.x env:
``
pip install timesfm[torch]
``
After that you can install the gift-eval package:
``
pip install -e .
``
