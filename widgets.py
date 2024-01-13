# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:07:36 2019

@author: Administrator
"""

from progressbar.widgets import *

class OtherInfo(Widget):
    
    def __init__(self, index=0):
        self.index = index
        
    def update(self, pbar):
        if self.index > len(pbar.other_info) - 1:
            return ''
        else:
            return '%s' % pbar.other_info[self.index]

class NumPercentage(Widget):
    
    def update(self, pbar):
        return '%d/%d' % (pbar.currval,pbar.maxval)
    
    
class Bar(WidgetHFill):
    """A progress bar which stretches to fill the line."""

    __slots__ = ('marker', 'left', 'right', 'fill', 'fill_left')

    def __init__(self, marker='#', left='|', right='|', fill=' ',
                 fill_left=True):
        """Creates a customizable progress bar.

        marker - string or updatable object to use as a marker
        left - string or updatable object to use as a left border
        right - string or updatable object to use as a right border
        fill - character to use for the empty part of the progress bar
        fill_left - whether to fill from the left or the right
        """
        self.marker = marker
        self.left = left
        self.right = right
        self.fill = fill
        self.fill_left = fill_left


    def update(self, pbar, width):
        """Updates the progress bar and its subcomponents."""

        left, marked, right = (format_updatable(i, pbar) for i in
                               (self.left, self.marker, self.right))

#        width -= len(left) + len(right)
        width = 20
        # Marked must *always* have length of 1
        if pbar.maxval is not UnknownLength and pbar.maxval:
          marked *= int(pbar.currval / pbar.maxval * width)
        else:
          marked = ''

        if self.fill_left:
            return '%s%s%s' % (left, marked.ljust(width, self.fill), right)
        else:
            return '%s%s%s' % (left, marked.rjust(width, self.fill), right)


