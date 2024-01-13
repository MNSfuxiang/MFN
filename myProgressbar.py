# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:01:39 2019

@author: Administrator
"""

from progressbar.progressbar import ProgressBar
from widgets import *

class MyProgressBar(ProgressBar):
    def __init__(self, maxval=None, widgets=None, term_width=None, poll=1,
                 left_justify=True, fd=None):
        self.other_info = []
        super(MyProgressBar, self).__init__(maxval,widgets,term_width,poll,left_justify,fd)
        
    
    def update(self, value=None, info=[]):
        self.other_info = list(info)
        try:
            super(MyProgressBar, self).update(value)
        except ValueError:
            print('Value out of range in progressbar')