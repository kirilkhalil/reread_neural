# reread_neural
## Neural Network for ReRead project

### Synopsis

First iterations are based on the work of [Dandurand et al. (2013)](https://www.tandfonline.com/doi/pdf/10.1080/09540091.2013.801934).  
This project aims first to develop a neural network implementation of the dual-stage view on visual word recognition 
(Hautala et al., 2021) and then to extend this model into continuous reading. This later goal will be enabled by 
implementing mechanisms of preview processing of upcoming word and forward shift in text input simulating forward saccade
length in reading. The architecture of the neural network for visual word recognition is expected to take a form of a 
hybrid autoencoder consisting of parallel processing encoding layers simulating orthographic processing and followed by
recurrent decoding processing layers simulating phonological decoding. The design of encoder network is based on earlier
work by Dandurand et al. (2013) and the design of decoding network is based on earlier work by Sibley et al. (2012). 
The architecture of the planned continuous reading model will build on theoretical understanding provided by existing 
integrative models (Snell et al., 2018; Li & Pollatsek, 2020) combining connectionist visual word recognition modules 
with eye movement control modules.  

The programming work will be conducted by relying on Keras -application programming 
interface for deep learning in Python -programming language. The work has been funded by grant 317030 from 
Academy of Finland to Jarkko Hautala. The site of the work is Niilo Mäki Institute, Jyväskylä, Finland. The programming 
work will be conducted by Kiril Khalil under supervision of Jarkko Hautala, and the team from the Faculty of Information
Technology at University of Jyväskylä consisting of Paavo Nieminen, Mirka Saarela and Tommi Kärkkäinen.

------------------------------------------------------------------------- 
-------------------------------------------------------------------------

#### For more information:  

Hautala, J., Hawelka, S., & Aro, M. (2021). Dual-stage and dual-deficit? Word recognition processes during text reading 
across the reading fluency continuum. Reading and Writing, 1-24. https://doi.org/10.1007/s11145-021-10201-1

Dandurand, F., Hannagan, T., & Grainger, J. (2013). Computational models of location-invariant orthographic processing. 
Connection Science, 25(1), 1-26.

Sibley, D. E., & Kello, C. T. (2012). Learned Orthographic Representations Facilitates Large-Scale Modeling of Word 
Recognition. In Visual Word Recognition Volume 1 (pp. 28-51). Psychology Press.

Snell, J., van Leipsig, S., Grainger, J., & Meeter, M. (2018). OB1-reader: A model of word recognition and eye movements
in text reading. Psychological review, 125(6), 969.

Li, X., & Pollatsek, A. (2020). An integrated model of word processing and eye-movement control during Chinese reading. 
Psychological Review, 127(6), 1139.

------------------------------------------------------------------------- 
-------------------------------------------------------------------------

### Implementation

Requirements:  

- Python 3.9.5   
- TensorFlow 2.11.0 (Keras should be bundled in the TF installation if not then Keras is required to be installed).
Recommend using Anaconda or Miniconda to handle TF installation process.  
- NumPy 1.24

------------------------------------------------------------------------- 
-------------------------------------------------------------------------

### Usage  

- 03.04.2023: Usage requires a corpus/library of words in a text file with each word being the same length with no 
special characters (including ä, ö , å). This is read as the input data. Working example to run is: 'zero_deck.py' for
the zero deck example.
- 24.04.2023: Current working script is: 'one_deck.py' and the required inputs are provided. 
'positional_supervised_corpus.rtf' for inputs and 'labels.rtf' for target outputs. You can make your own similar data
by using the: 'mod_inputs.py' -file to create inputs and target outputs from a corpus with the restrictions mentioned
in the 03.04.2023 entry.
- 12.05.2023: When the file 'two_deck.py' is run it saves the trained model as 'lower_deck.h5' and the mapping used as
'lower_deck_mapping.pkl'. After training you can evaluate the outputs of the lower part of the 'two-deck-topology' by
running the file 'lower_deck_evaluation.py' and changing the test input accordingly (or feeding a list of inputs).

------------------------------------------------------------------------- 
-------------------------------------------------------------------------

### Project evolution

- 01.03.2023: Project started on/off. Basic Keras/TF tutorials and theory regarding NeuralNets
- 03.04.2023: Working implementation of 'zero-deck-topology'. Prep work for one-deck-topology.
- 24.04.2023: Preliminary implementation of 'one-deck-topology'.
- 12.05.2023: Lower deck of 'two-deck-topology' produces wanted outputs.