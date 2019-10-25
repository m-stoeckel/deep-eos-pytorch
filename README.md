# Deep-EOS PyTorch
## Introduction
This is a pytorch implementation of "Deep-EOS: General-Purpose Neural Networks for Sentence Boundary Detection" (Schweter et al, 2019).
Go here for the original Keras implementation: [https://github.com/stefan-it/deep-eos](https://github.com/stefan-it/deep-eos)

Currently only the LSTM and Bi-LSTM models are implemented.

The dataset download script and dataset creation methods are largely copied from Stefan Schweters repository. 

# Results
## Test set
| Language     | PyTorch LSTM | PyTorch Bi-LSTM | Original LSTM   | Original Bi-LSTM   | Original CNN   |
| ------------ | ------------ | --------------- | --------------- | ------------------ | -------------- |
| German       | **0.9763**   | **0.9763**      | 0.9750          | 0.9760             | 0.9751         |
| English      | **0.9862**   | **0.9862**      | 0.9861          | 0.9860             | 0.9858         |
| Bulgarian    | 0.9893       | 0.9891          | 0.9922          | **0.9923**         | 0.9919         |
| Bosnian      | 0.9917       | 0.9919          | 0.9957          | **0.9959**         | 0.9953         |
| Greek        | 0.9958       | 0.9957          | 0.9967          | **0.9969**         | 0.9963         |
| Croatian     | 0.9921       | 0.9918          | 0.9946          | **0.9948**         | 0.9943         |
| Macedonian   | 0.9772       | 0.9786          | 0.9810          | **0.9811**         | 0.9794         |
| Romanian     | 0.9875       | 0.9874          | **0.9907**      | 0.9906             | 0.9904         |
| Albanian     | 0.9922       | 0.9920          | **0.9953**      | 0.9949             | 0.9940         |
| Serbian      | 0.9843       | 0.9838          | **0.9877**      | **0.9877**         | 0.9870         |
| Turkish      | 0.9824       | 0.9829          | **0.9858**      | 0.9854             | 0.9854         |

## Download
The trained models can be downloaded in the releases section.
The each model 7zip archive contains the best scoring model (on development data) out of 5 epochs,
alongside with its corresponding vocabulary.

The datasets can be downloaded with the ``download_data.sh`` script from the original implementation. 

# To-Do
- Implementation:
    - [x] LSTM
    - [x] Bi-LSTM
    - [ ] CNN
    
# References
- S. Schweter and S. Ahmed, "Deep-EOS: General-Purpose Neural Networks for Sentence Boundary Detection” in Proceedings of the 15th Conference on Natural Language Processing (KONVENS), 2019.