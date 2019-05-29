# GDGAN.pytorch
A PyTorch implementation of https://arxiv.org/abs/1812.01690

## Dependencies
- PyTorch 1.0.0 or later
- numpy
- scipy
- sklearn
- [tensorboardX](https://github.com/lanpa/tensorboardX)
  - see the link for more dependencies

## Train

### GDGAN
```
$ python main.py
```
See possible arguments by `$ python main.py --help`

### Classifier
```
$ python classifier/classifier.py
```
See possible arguments by `$ python classifier/classifier.py --help`

## Loss (hyperparameters detail)
<img src="https://latex.codecogs.com/gif.latex?L^{G1}&space;=&space;L^{G}_{WGAN\_GP}&space;&plus;&space;\lambda_{1}^{G1}&space;BCE(D_{c}^{(1)}&space;(G^{(1)}(z,y^{(1)})),&space;y^{(1)})"/>
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;L^{D1}&space;=&space;L^{D}_{WGAN\_GP}&space;&&plus;&space;\lambda_{1}^{D1}&space;BCE(D_{c}^{(1)}&space;(G^{(1)}(z,y^{(1)})),&space;y^{(1)})&space;\\&&plus;&space;\lambda_{2}^{D1}&space;BCE(D_{c}^{(1)}&space;(x),&space;y^{(1)})&space;\end{align*}"\>
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;L^{G2}&space;=&space;L^{G}_{WGAN\_GP}&space;&&plus;&space;\lambda_{1}^{G2}BCE(D_{c}^{(2)}(G^{(2)}(G^{(1)}(z,y^{(1)}),&space;y^{(2)})),&space;y^{(2)})&space;\\&space;&&plus;&space;\lambda_{2}^{G2}&space;BCE(D_{c}^{(1)}(G^{(2)}(G^{(1)}(z,y^{(1)}),&space;y^{(2)})),&space;y^{(1)})&space;\\&space;&&plus;&space;\lambda_{3}^{G2}&space;MSE&space;(G^{(1)}(z,&space;y^{(1)}),&space;G^{(2)}(G^{(1)}(z,y^{(1)}),&space;y^{(2)}))&space;\end{align*}"\>
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;L^{D2}&space;=&space;L^{D}_{WGAN\_GP}&space;&&plus;&space;\lambda_{1}^{D2}BCE(D_{c}^{(2)}(G^{(2)}(G^{(1)}(z,y^{(1)}),&space;y^{(2)})),&space;y^{(2)})&space;\\&space;&&plus;&space;\lambda_{2}^{D2}&space;BCE&space;(D_{c}^{(2)}(x),&space;y^{(2)})&space;\\&space;&&plus;&space;\lambda_{3}^{D2}&space;BCE&space;(D_{c}^{(1)}(G^{(2)}(G^{(1)}(z,y^{(1)}),&space;y^{(2)})),&space;y^{(1)})&space;\end{align*}"\>
