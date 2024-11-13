# ssnet-python
State Space Neural Networks - A PyTorch-powered interface to perform NN-based system identification

## Usage
For more information on this code, please refer to my PhD dissertation (in particular, to Chapter 3 and Chapter 4)

> Fabio Bonassi, “_Reconciling deep learning and control theory: recurrent neural networks for model-based control design_,” 2023, Politecnico di Milano. PhD Dissertation. Supervisor: Prof. Riccardo Scattolini, Prof. Marcello Farina [[link](https://www.politesi.polimi.it/handle/10589/196384)]

If you use the code, please consider citing the PhD dissertation. 
```
@phdthesis{bonassi2023reconciling,
  title = {Reconciling deep learning and control theory: recurrent neural networks for model-based control design},
  author = {Bonassi, Fabio},
  year = {2023},
  month = feb,
  address = {Milan, Italy},
  school = {Politecnico di Milano},
  type = {PhD thesis},
}
```

Or, alternatively, the corresponding Springer Brief
```
@incollection{bonassi2024reconciling,
  title = {Reconciling Deep Learning and Control Theory: Recurrent Neural Networks for Indirect Data-Driven Control},
  author = {Bonassi, Fabio},
  booktitle = {Special Topics in Information Technology},
  pages = {77--87},
  year = {2024},
  publisher = {Springer},
  doi = {10.1007/978-3-031-51500-2_7},
}
```

## Example
To illustrate how to use this python library, the script `example_ph.py` is now included.
The script fits a stable deep GRU to the data collected from the pH neutralization process.

## Requirements
You can install the requirements by running `pip install -r requirements.txt`. You need Python 3.10 or later.