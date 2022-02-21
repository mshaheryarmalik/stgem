#!/usr/bin/python3
# -*- coding: utf-8 -*-

from stgem.sut import SUT

from stgem import load_stgem_module
def loadSUT(name):
   return load_stgem_module(name,"sut")

