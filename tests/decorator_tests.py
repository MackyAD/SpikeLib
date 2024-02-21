#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:17:51 2024

@author: marcos
"""

def class_register(cls):
    # define what categories we want
    # we then need to define a decorator for each one
    categories = 'foo', 'bar'
    
    # add empty lists for each category
    for category in categories:
        setattr(cls, f'_{category}', [])
    
    # scan all methods to see which one have been registered
    for methodname in dir(cls):
        method = getattr(cls, methodname)
        
        for category in categories:
            if hasattr(method, f'_{category}'):
                getattr(cls, f'_{category}').append(methodname)
            
    return cls
    
# Define decorators
def foo(func):
    func._foo = None
    return func

def bar(func):
    func._bar = None
    return func

# Define the decorated class with decorated methods
@class_register
class MyClass:
        
    def __init__(self):
        self._bar = []
        self._foo = []

    @foo            
    def add(self, a, b):
        return a+b
    
    def sub(self, a, b):
        return a-b
    
    @foo
    @bar
    def concat(self, *a):
        return '; '.join(str(x) for x in a)
    
mc = MyClass()
print(mc._foo)
#%%
def class_register_categories(cls):
    categories = 'foo', 'bar'
    
    for category in categories:
        setattr(cls, category, Category())
    
    for methodname in dir(cls):
        method = getattr(cls, methodname)
        for category in categories:
            if hasattr(method, '_'+category):
                getattr(cls, category).append(methodname)
        
    return cls

class Category(list):
    
    def __repr__(self):
        string = '\n'.join(self)
        return string

def foo(func):
    func._foo = None
    return func

def bar(func):
    func._bar = None
    return func


@class_register_categories
class MyClass(object):

    @foo
    def my_method(self, arg1, arg2):
        pass
    
    @foo
    @bar
    def my_other_method(self, arg1, arg2):
        pass
    
    def yet_another_method(self):
        pass
    
myclass = MyClass()

for obj in (myclass, MyClass):
    print('foo:\n', obj.foo)
    print('bar:\n', obj.bar)
    print()
        
#%%

def dec(func):
    
    def inner(*a, **kw):
        print('Running', func.__name__)
        
        return func(*a, **kw)
    return inner

@dec
def foo(a):
    print(f'========{a}========')
    
#%%

import functools

def enter_exit_info(func):
    @functools.wraps(func)
    def wrapper(self, *arg, **kw):
        print('-- entering', func.__name__)
        print('-- ', self.__dict__)
        res = func(self, *arg, **kw)
        print('-- exiting', func.__name__)
        print('-- ', self.__dict__)
        return res
    return wrapper

class TestWrapper():
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.c = 0
    
    @enter_exit_info
    def add_in_c(self):
        self.c = self.a + self.b
        print(self.c)

    @enter_exit_info
    def mult_in_c(self):
        self.c = self.a * self.b
        print(self.c)


t = TestWrapper(2, 3)
t.add_in_c()
# t.mult_in_c()

#%%

def category(name: str, /):
    def decorator(func):
        try:
            func._categories.add(name)
        except AttributeError:
            func._categories = {name}
        return func
    return decorator

def register_categories(cls):
    categories: dict[str, set[str]] = {}
    # Get all methods and their categories
    for name, method in cls.__dict__.items():
        if hasattr(method, '_categories'):
            for category in method._categories:
                categories.setdefault(category, Category()).add(name)

    # Create the category attributes from the aggregated categories
    for category, methods in categories.items():
        setattr(cls, category, methods)

    return cls

class Category(set):
    def __repr__(self):
        string = '\n'.join(self)
        return string

foo = category('foo')
bar = category('bar')

@register_categories
class A:

    @foo
    def a(self):
        pass

    def b(self):
        pass
    
    @foo
    @bar
    def c(self):
        pass
    
a = A()
print(a.foo)
print(a.bar)