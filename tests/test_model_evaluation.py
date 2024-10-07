import os
import sys
import pytest
import unittest
from pytest import approx

#append the relative path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model_evaluation import evaluate_model

def test_evaluate_model():
    pass  
