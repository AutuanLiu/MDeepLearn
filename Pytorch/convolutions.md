# 原理可视化: [Setosa data visualization and visual explanations](http://setosa.io/#/)
## image kernel: [Image Kernels explained visually](http://setosa.io/ev/image-kernels/)
* The blur kernel de-emphasizes differences in adjacent pixel values
    ```
    0.0625 0.125 0.0625
    0.125  0.25  0.125
    0.0625 0.125 0.0625
    ```
* sobel kernels are used to show only the differences in adjacent pixel values in a particular direction.
    ```
    -1 -2 -1
     0  0  0
     1  2  1
    ```
* The emboss kernel (similar to the sobel kernel and sometimes referred to mean the same) givens the illusion of depth by emphasizing the differences of pixels in a given direction. In this case, in a direction along a line from the top left to the bottom right.
    ```
    -2  -2  0
    -1   1  1
     0   1  2
    ```
* The indentity kernel leaves the image unchanged.
    ```
     0  0  0
     0  1  0
     0  0  0
    ```
* An outline kernel (also called an "edge" kernel) is used to highlight large differences in pixel values. A pixel next to neighbor pixels with close to the same intensity will appear black in the new image while one next to neighbor pixels that differ strongly will appear white.
    ```
    -1 -1 -1
    -1  8 -1
    -1  1  2
    ```
* The sharpen kernel emphasizes differences in adjacent pixel values. This makes the image look more vivid.
    ```
     0 -1  0
    -1  5 -1
     0 -1  0
    ```

## [Eigenvectors and Eigenvalues explained visually](http://setosa.io/ev/eigenvectors-and-eigenvalues/)
## [Markov Chains explained visually](http://setosa.io/ev/markov-chains/)
## [Simpson's Paradox](http://vudlab.com/simpsons/)
## [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/chap4.html)