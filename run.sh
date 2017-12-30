#!/usr/bin/env bash

# an example for running python scripts on terminal
# for env test and make sure all the dependencies are available

# pytorch test
cd Pytorch
python LinearSimple.py
echo "------------------"
echo "-- pytorch pass --"
echo "------------------"
cd ..

# imbalanceLearn test
cd imbalanceLearn
python combination.py
echo "-------------------------"
echo "-- imbalanceLearn pass --"
echo "-------------------------"
cd ..

# sklearn test
cd sklearn
python SVR.py
echo "------------------"
echo "-- sklearn pass --"
echo "------------------"
cd ..

#tensorflow test
cd TensorFlow
python test.py
echo "---------------------"
echo "-- TensorFlow pass --"
echo "---------------------"
cd ..

# tensorlayer test
cd TensorLayer
python mnist_simple.py
echo "----------------------"
echo "-- TensorLayer pass --"
echo "----------------------"
cd ..

# xgboost test
cd XGBoost
python xgboostree.py
echo "------------------"
echo "-- XGBoost pass --"
echo "------------------"
cd ..

echo "-----------------"
echo "-- @All pass!@ --"
echo "-----------------"