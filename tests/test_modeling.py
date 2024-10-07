import os 
import sys
import pytest
import unittest
from pytest import approx
#append the relative path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from modeling import (
    train_and_evaluate_logistic_regression,
    train_and_evaluate_random_forest,
    train_and_evaluate_xgboost,
    train_and_evaluate_adaboost,
    train_and_evaluate_decision_tree
)

def test_train_and_evaluate_logistic_regression():
    pass  

def test_train_and_evaluate_random_forest():
    pass  

def test_train_and_evaluate_xgboost():
    pass  

def test_train_and_evaluate_adaboost():
    pass  

def test_train_and_evaluate_decision_tree():
    pass  
