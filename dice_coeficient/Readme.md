# Dice coeficient calculator
Dice coefficient between two boolean NumPy arrays or array-like data. This is commonly used as a set similarity measurement (though note it is not a true metric; it does not satisfy the triangle inequality). The dimensionality of the input is completely arbitrary, but `im1.shape` and `im2.shape` much be equal. 

This code is based on the function developed [here](https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137) by [Bruno Guberfain do Amaral](https://gist.github.com/brunodoamaral).

## Usage
```sh
python dice.sh img1 img2 {invert1} {invert2}
```

where `invert1` and `invert2` are optional parameters used in case of the images have inverted colors.