from copy import deepcopy
import operator as op
from functools import reduce
import re
import os
import glob
import time
import secrets
import string
import hashlib
import csv

from itertools import permutations
from datetime import datetime
from typing import Type

from utils.array_utilities import *




'''

Outline:






'''



'''

Type Operations

'''

def isinstances(obj, types):
    for type_ in types:
        if isinstance(obj, type_): return True
    return False


''' will check if input "func" is a lambda function '''
def islambda(func):
    if callable(func):
        if func.__name__ == "<lambda>":
            return True
    return False


''' will return the number of input arguments of a lambda func '''
def lambda_argsCount(func):
    if islambda(func):
        return func.__code__.co_argcount
    else: raise TypeError('Input "func" must be a lambda function.')


'''

Time Constructs

'''

class timer():
    def __init__(self,interval=None):
        if interval==None:
            self.interval=None
        elif isinstance(interval,tuple(*int_types,*float_types)) and interval > 0:
            self.start = float(time.time())
            self.end = self.start + interval
        else:
            print("Invalid entry. Only one optional input ('time') is allowed and has to be an int or float.")

    def tic(self):
        self.start = float(time.time())
        #unsetting end-time if previously set by 'interval' OR 'toc'.
        self.end = None

    def toc(self):
        self.end = float(time.time())
        if self.interval == None:
            self.elapsed = self.end - self.start


'''

Exception Handling

'''

# def assert_items_instance(items, instance, cur_items=None):
#     if cur_items!=None: 
#     for item in items


'''

Combinatorics

'''

def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  


def combination(i,l,r,chk=None):
    n=len(l)
    if chk:
        if i>=nCr(n, r): raise IndexError('index out of range.')
    ncr = 1
    for n_i, i_1 in zip(range(n, n - r, -1), range(1, r + 1)):
        ncr *= n_i
        ncr //= i_1
    c_i = ncr
    for r in range(r, 0, -1):
        ncr *= r
        ncr //= n
        while c_i - ncr > i:
            c_i -= ncr
            ncr *= (n - r)
            ncr -= ncr % r
            n -= 1
            ncr //= n
        n -= 1
        yield l[n]

'''

Useful Transfer Funcs

'''

def rectify(x):
    return x if x>=0 else 0

'''

Random Generators

'''

# Generate random point in range
''' commented-out checks needs replacement; prev. archived '''
def rrand(start, end, chk=None):
    # if chk:
    #     if not are_nums(start,end): raise TypeError('Inputs must be "num-types".')

    return float(np.random.rand(1)*abs(end - start) + start)

# Generate an array of random points in range
''' commented-out checks needs replacement; prev. archived '''
def rrand_gen(start,end,n,chk=None):
    # if chk:
    #     try:
    #         if not are_nums(start,end) or not are_positive_int(n): raise Exception
    #     except: raise TypeError('"start" and "end" must be "num-types"."n" must be a positive-int.')

    for _ in range(n): yield rrand(start,end)

''' commented-out checks needs replacement; prev. archived '''
def rrand_vec(start,end,n,chk=None):
    # if chk:
    #     try:
    #         if not are_nums(start,end) or not are_positive_int(n): raise Exception
    #     except: raise TypeError('"start" and "end" must be "num-types"."n" must be a positive-int.')

    return np.random.rand(n)*abs(end - start) + start


''' will return a random integer with N digits '''
def randomDigits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return np.random.randint(range_start, range_end)


''' will return a random string with N characters (any ascii) '''
def randomString(n):
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(n))



def qrand_gen(n):
    if isinstance(n,int):
        return np.round(n*np.random.rand(1)+0.5)-1
    else:
        print('Input must be an "int".')


'''

Cryptography

'''

''' sha256 encoding of a string-key '''
def sha256_encode(key):
    if isinstance(key, str): return hashlib.sha256(key.encode()).hexdigest()
    else: raise TypeError('Input must be a string.')


''' Verifying if string key belongs to the same sha256 "hash_" (checking with "hexdigest" of it) '''
def sha256_verify(hash_, key):
    if not isinstance(hash_, str): raise TypeError('"hash_" must be a "str" (hexdigest of sha256 function)')
    if not isinstance(key, str): raise TypeError('"key" must be a "str".')

    return hash_ == hashlib.sha256(key.encode()).hexdigest()


'''

List Operations

'''

''' Will check if at least one object is the valid type'''
def at_least_one(type_, *objs):
    for obj in objs: 
        if isinstance(obj, type_): return True
    return False


# Will return True if has duplicates, else false
def has_duplicates(l):
    if not isinstance(l, list): raise TypeError('Input must be a list.')
    return len(l)!=len(set(l))


# Remove all duplicates from list (if order not a concern)
def remove_all_duplicates(l):
    if not isinstance(l, list): raise TypeError('Input must be a list.')
    return list(set(l))

# Delete all empty lines:
remove_empty = lambda lines: filter(lambda x: not re.match(r'^\s*$', x), lines)


def lists_concat(*arrs,chk=None):
    if chk:
        for arr in arrs:
            if not isinstance(arr,list): raise TypeError('All inputs must be lists.')

    new_arr = []
    for arr in arrs: new_arr += arr
    return new_arr




# Insert element in given list in sorted position
# assumes no duplicates... gives option of ref sorting, see function below...
# if duplicate is entered... does nothing
# needs chk
def insert_sorted(arr, n):
      
    for i in range(len(arr)):
        if arr[i]==n: return arr           
        elif arr[i]>n: break

    return arr[:i] + [n] + arr[i:]

# Insert element in given list based on its counterpart's
# sorted position in counterpart list
# assumes no duplicates...
# needs chk
def insert_ref_sorted(arr,n,ref_arr,el):
    for i in range(len(arr)):
        if arr[i]==n: return [arr,ref_arr]           
        elif arr[i]>n: break

    return [arr[:i] + [n] + arr[i:],ref_arr[:i] + [el] + ref_arr[i:]]


# insert a list of durations, and output a list of start indices 
# for those durations as if their lists were concatenated
def durs_to_starts(durs):
    strts,acc=[],0
    for dur in durs:
        strts.append(acc)
        acc+=dur
    return strts

# insert a list of durations, and output a list of end indices 
# for those durations as if their lists were concatenated
def durs_to_ends(durs):
    ends,acc=[],0
    for dur in durs:
        acc+=dur
        ends.append(acc)
    return ends


# insert a list of durations, and output a list of start and end indices 
# for those durations as if their lists were concatenated
def durs_to_starts_ends(durs):
    inds,srt_acc,end_acc=[],0,0
    for dur in durs:
        end_acc+=dur
        inds.append([srt_acc,end_acc])
        srt_acc+=dur
    return inds


# let M be an m x n 2-dimensional list
# needs chk
def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]



# let M be an m x n 2-dimensional list
# needs chk
# will sort columns based on ith row, reverse=True will make it descending order
'''
This function was verified on 11/30/2021, see test macro: sort_by_row_11_30_2021.txt
'''
def sort_by_row(M,ind,reverse=False): 
    if ind<0 or ind>len(M[0]): 
        raise ValueError('ind must be a zero or positive integer <= n where n is number of columns')

    return transpose(sorted(transpose(M),key=lambda l:l[ind], reverse = reverse))



# let M be an m x n 2-dimensional list
# needs chk
# will sort row based on ith column , reverse=True will make it descending order
'''
This function was verified on 11/30/2021, see test macro: sort_by_col_11_30_2021.txt
'''
def sort_by_col(M,ind,reverse=False): 
    if ind<0 or ind>len(M): 
        raise ValueError('ind must be a zero or positive integer <= m where m is number of rows')

    return sorted(M,key=lambda l:l[ind], reverse=reverse)


def append_items(arr,*itms):
    # create input check for itms later
    if isinstance(arr, np.ndarray):
        arr = list(arr)
        for itm in itms:
            arr.append(itm)
        return np.array(arr)
    elif isinstance(arr, list):
        for itm in itms:
            arr.append(itm)
        return arr
    else:
        # Raise Error Later.
        print('Invalid input. itms unpacked must be floats or ints.')  

def concatenate(*arrs):
    # create input check for size consistency later. Input check for arrs inside 'append_items()'
    tmp = []
    for arr in arrs:
        tmp = append_items(tmp,*arr)
    return tmp


def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


def incircle(px,py,cx,cy,r):
    if distance(px,py,cx,cy) < r:
        return True
    else:
        return False
 

def inrect(px,py,x1,y1,x2,y2):
    if (px >= x1 and px <= x2) and (py >= y1 and py <= y2) :
        return True
    else:
        return False  


# flatten 2nd order nested list
def flatten_nest(ls):
    return [i for l in ls for i in l]


# will check if all lists input are the same length or not
def are_same_len(*lists_, chk = False):
    if chk:
        for list_ in lists_:
            if not isinstance(list_, list): 
                raise TypeError('all lists_ must be list objects.')

    len0 = len(lists_[0])
    for list_ in lists_[1:]:
        if len(list_) != len0: return False
    
    return True


def is_ordered_sequence(*v):
    # make sure they are positive integers
    # sort them
    # make sure there are no duplicates
    # make sure there are no gaps

    pass


def in_list(val_, list_, chk=False):
    if chk:
        if not isinstance(list_, list):
            raise TypeError('list_ must be a list object')
        
    if val_ in list_: return True
    else: return False


'''

String Operations

'''

lowerbets=r'abcdefghijklmnopqrstuvwxyz'
upperbets=r'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
numerics =r'0123456789'
specials = '''`~!@#$%^&*()|\-_=+{[}];:'"<,>.?/''' # more on this...
alphabets=lowerbets+upperbets
alphanumerics = alphabets+numerics
english = alphanumerics+specials


'''
Will return the string of index i with fixed length (len_).
If index is smaller than len_, it will be padded with zeros (to the left) to equal len_
and if it is larger than len_, it will be truncated (from the left) to equal len_
'''
def fixed_size_index(i, len_):
    ref=len_*'0'
    str_i = ref+str(i)
    return str_i[len(str_i)-len_:]



# Takes any number of strings and concatenates them together into one string
# Has a chk to make sure inputs are 'str' type.
'''
This function was verified on 12/14/2021, see test macro
'''
def str_concat(*strings,chk=None):
    if chk:
        for string in strings:
            if not isinstance(string,str): raise TypeError('All inputs must be str-types.')

    new_string = ''
    for string in strings: new_string += string
    return new_string


# Takes any raw string that has only characters from 'english' string and white-spaces 
# (will raise error otherwise) and splits it at the 'english' characters (ignores white-spaces)
# and returns list of these chars.
'''
This function was verified on 12/14/2021, see test macro: split_string_english_12_14_2021.txt
'''
def split_string_english(string):
    if not isinstance(string, str):
        raise TypeError('Input must be a "string" object.')

    string = str_concat(*string.split())

    for char in string:
        if not char in english: 
            raise ValueError('Input string must consist of only white-spaces or english characters.') 
        yield char


# add chk later
def start_inds(stream,*phrases):
    inds=[]
    for phrase in phrases:
        inds.append([m.start() for m in re.finditer(phrase, stream)])
    inds.sort()
    inds = lists_concat(*inds)
    return inds

# add chk later
def end_inds(stream,*phrases):
    inds=[]
    for phrase in phrases:
        inds.append([m.end() for m in re.finditer(phrase, stream)])
    inds.sort()
    inds = lists_concat(*inds)
    return inds

# NEEDS TESTING
# will extract from stream starting at an index up until stream 
# has any one of the characters from matches.
# add chk later
def til_matches(ind, stream, matches):
    cond = True
    extract=[]
    while cond and ind<len(stream):
        if stream[ind] in matches: cond=False 
        else: 
            extract.append(stream[ind])
            ind+=1
    if extract: return str_concat(*extract)



# will extract from stream starting at an index up until stream 
# doesn't have any one of the characters from matches.
# add chk later
def til_unmatches(ind, stream, matches):
    cond = True
    extract=[]
    while cond and ind<len(stream):
        if stream[ind] in matches: 
            extract.append(stream[ind])
            ind+=1
        else: cond=False
    if extract: return str_concat(*extract)


# has at least one symbol from chars
def has_chars(inputs,chars):
    for symbol in inputs: 
        if symbol in chars: return True
    return False

    

# has only symbols from chars
def has_only_chars(inputs,chars):
    for symbol in inputs:
        if symbol not in chars: return False
    return True

# has all the symbol from chars
def has_all_chars(inputs,chars):
    for char in chars:
        if char not in inputs: return False
    return True

# has only all the symbols from chars
def has_only_all_chars(inputs,chars):
    return has_only_chars(inputs,chars) and has_all_chars(inputs,chars)

def remove_chars(inputs,chars):
    return re.sub(chars, '', inputs)

def replace_chars_with_char(in_,chars,char):
    for chr in chars: in_=in_.replace(chr,char)
    return in_

def only_chars(inputs,chars):
    return (symb for symb in inputs if symb in chars)



# will take a string (in lines)
# and convert to a list of those line strings
def lines_to_list(lines_):
    return lines_.splitlines()

# will take a list of strings
# and join them together as lines
def list_to_lines(list_):
    lines_=''
    for line_ in list_: lines_+=(line_+'\n')
    return lines_

# input must be a string (needs a chk...)
# will insert line at a given line number (positive integer 1,2,3, ...)
# if no input will insert at end

'''
This function was verified on 11/30/2021, see test macro: insert_line_11_30_2021.txt
'''
def insert_line(lines_,line_,ind=None):
    list_ = lines_to_list(lines_)
    if ind==None:
        return list_to_lines([*list_,line_]) #adding line at end
    else:
        if ind<1 or ind>len(list_)+1: 
            raise ValueError('ind must be a positive integer <= N+1 where N is number of lines')

        ind-=1
        try: 
            list_ = [*list_[:ind],line_,*list_[ind:]] 
        except: raise ValueError('ind must be a positive integer <= N+1 where N is number of lines')
        return list_to_lines(list_)

# input must be a string (needs a chk...)
# will insert lines at given line numbers (positive integer 1,2,3, ...)
# lines_inds will be Nx2 list where N is number of lines to insert,
# first element is line string, and second element is the target line number (need check for this)
'''
This function was verified on 11/30/2021, see test macro: insert_lines_11_30_2021.txt
'''
def insert_lines(lines_,lines_inds):
    list_ = lines_to_list(lines_)
    for line_ind in lines_inds:
        if line_ind[1]<1 or line_ind[1]>len(list_)+1: 
            raise ValueError('ind must be a positive integer <= N+1 where N is number of lines')

    lines_inds = sort_by_col(lines_inds, 1) # sort by inds

    ct=0
    for line_ind in lines_inds:
        [line_,ind] = line_ind
        ind-=1
        try: 
            list_ = [*list_[:ind+ct],line_,*list_[ind+ct:]] 
        except: raise ValueError('ind must be a positive integer <= N+1 where N is number of lines')
        ct+=1
    return list_to_lines(list_)


# input must be a string (needs a chk...)
# will remove line at a given line number (positive integer 1,2,3, ...)
# if no input will remove last line

'''
This function was verified on 11/30/2021, see test macro: remove_line_11_30_2021.txt
'''
def remove_line(lines_,ind=None):
    list_ = lines_to_list(lines_)
    if ind==None:
        return list_to_lines(list_[:len(list_)-1]) #removing last line
    else:
        if ind<1 or ind>len(list_): 
            raise ValueError('ind must be a positive integer <= N where N is number of lines')

        ind-=1
        try: list_ = [*list_[:ind],*list_[ind+1:]] 
        except: raise ValueError('ind must be a positive integer <= N where N is number of lines')
        return list_to_lines(list_)


# input must be a string (needs a chk...)
# will remove lines at given line numbers (positive integer 1,2,3, ...)
# inds will be list of target line numbers
'''
This function was verified on 11/30/2021, see test macro: remove_lines_11_30_2021.txt
For some reason it will still execut if 'inds'=[], just don't accidentally put this...
'''
def remove_lines(lines_,inds):
    list_ = lines_to_list(lines_)
    for ind in inds:
        if ind<1 or ind>len(list_): 
            raise ValueError('ind must be a positive integer <= N where N is number of lines')

    inds.sort() # sort by inds

    ct=0
    for ind in inds:
        ind-=1
        try: list_ = [*list_[:ind+ct],*list_[ind+1+ct:]] 
        except: raise ValueError('ind must be a positive integer <= N where N is number of lines')
        ct-=1
    return list_to_lines(list_)




# input must be a string (needs a chk...)
# will remove line at a given line number (positive integer 1,2,3, ...)
# if no input will replace last line

'''
This function was verified on 11/30/2021, see test macro: replace_line_11_30_2021.txt
'''
def replace_line(lines_,line_,ind=None):
    list_ = lines_to_list(lines_)
    if ind==None:
        return list_to_lines([*list_[:len(list_)-1],line_]) #replacing line at end
    else:
        if ind<1 or ind>len(list_): 
            raise ValueError('ind must be a positive integer <= N where N is number of lines')

        ind-=1
        try: 
            list_ = [*list_[:ind],line_,*list_[ind+1:]] 
        except: raise ValueError('ind must be a positive integer <= N where N is number of lines')
        return list_to_lines(list_)


# input must be a string (needs a chk...)
# will replace lines at given line numbers (positive integer 1,2,3, ...)
# lines_inds will be Nx2 list where N is number of lines to insert,
# first element is line string, and second element is the target line number (need check for this)
'''
This function was verified on 11/30/2021, see test macro: replace_lines_11_30_2021.txt
'''
def replace_lines(lines_,lines_inds):
    list_ = lines_to_list(lines_)
    for line_ind in lines_inds:
        if line_ind[1]<1 or line_ind[1]>len(list_): 
            raise ValueError('ind must be a positive integer <= N where N is number of lines')

    lines_inds = sort_by_col(lines_inds, 1) # sort by inds

    for line_ind in lines_inds:
        [line_,ind] = line_ind
        ind-=1
        try: 
            list_ = [*list_[:ind],line_,*list_[ind+1:]] 
        except: raise ValueError('ind must be a positive integer <= N where N is number of lines')
    return list_to_lines(list_)

# function that will take any string and switch the back slashes to forward slashes (must enter raw string) 
# by default will append a forward slash at the end (since this is the more common use-case)
def to_forwardslashes(string_,endslash=True):
    if endslash: return string_.replace("\\",'/')+'/'
    else: return string_.replace("\\",'/')


''' will convert list of objects into a sentence of their strings seperated by commas, where final has "and" prior to '''
def comma_and_list(l):
    if len(l)==1:
        return str(l[0])
    elif len(l)==2:
        return str(l[0])+' & '+str(l[1])
    else:
        sent_ = ''
        for k, obj in enumerate(l):
            if k==0: sent_=sent_+str(obj)
            elif k==len(l)-1: sent_= sent_+', & '+ str(obj)
            else: sent_= sent_+', '+ str(obj)
        return sent_


'''

Dictionary Operations

    Helpers:
    - Both dict_[ * invalid key string *] and dict_[ * not string * ] will raise "KeyError"
    - To replace/add simultaneously, use dictionary's built-in update method

    Verification Log:
    - Following functions verified 2/8/2022 (see macro "dict_operations_2_8_2022:):
        objs_to_dict
        add_to_dict
        remove_from_dict
        replace_in_dict
    - Following functions verified 2/21/2022:
        has_key (see macro "has_key_2_21_2022")
        get_key (see macro "get_key_2_21_2022")

'''

# Method to unpack a dictionary value (put here for reference)
#
# *dict_.values()

# Will take in a dictionary(s) (dicts_) or unpacked dictionary(s) (kwargs), 
# and set those keys as attributes of obj_
def keys_to_attrs(obj_,*dicts_,**kwargs):
    for dict_ in dicts_:
        for key in dict_:
            setattr(obj_, key, dict_[key])
    for key in kwargs.keys():
        setattr(obj_, key, kwargs[key])


''' Check if key exists in dictionary '''
def has_key(dict_, key_, chk=True):
    if chk:
        if not isinstance(dict_, dict): raise TypeError("dict_ must be a dictionary.")
        if not isinstance(key_, str): raise TypeError("key_ must be a string.")

    if key_ in dict_.keys(): return True
    else: return False


''' will return first instance of key with given val '''
def get_key(dict_, val_, chk=True):
    if chk:
        if not isinstance(dict_, dict): raise TypeError("dict_ must be a dictionary.")

    for key, value in dict_.items():
         if val_ == value:
             return key
 
    raise Exception("no key exists for given val_")


''' will put objs into a dictionary where the keys are the indices in dictionary as strings '''
def objs_to_dict(*objs, prefix=None):
    dict_={}

    if isinstance(prefix, str):
        for k, obj in enumerate(objs): dict_[prefix+str(k)] = obj
    elif prefix==None:
        for k, obj in enumerate(objs): dict_[str(k)] = obj
    else: raise TypeError("If prefix is specified, it must be a string.")

    return dict_


''' will add new dictionary (objs) into existing dictionary where the keys cannot overlap '''
def add_to_dict(dict_, objs, chk=True):
    if chk:
        if not isinstance(dict_, dict): raise TypeError("dict_ must be a dictionary.")
        if not isinstance(objs, dict): raise TypeError("objs must be a dictionary.")
    
    for key_ in objs.keys():
        if key_ in dict_.keys(): raise KeyError('One or more keys already exists.')

    dict_.update(objs)


''' will remove objs from dict with specific said keys '''
def remove_from_dict(dict_, *keys, chk=True):
    if chk:
        if not isinstance(dict_, dict): raise TypeError("dict_ must be a dictionary.")

    for key in keys:
        try: dict_.pop(key)
        except: raise KeyError("One or more keys do not exist or are not strings.")


''' will replace part of dictionary overlapping in keys with objs with objs.
    If objs has keys not in dict_, throw error '''
def replace_in_dict(dict_, objs, chk=True):
    if chk:
        if not isinstance(dict_, dict): raise TypeError("dict_ must be a dictionary.")
        if not isinstance(objs, dict): raise TypeError("objs must be a dictionary.")
    
    for key_ in objs.keys():
        if not key_ in dict_.keys(): raise KeyError('One or more keys do not exist.')

    dict_.update(objs)

''' convert dictionary values to list '''
def vals_to_list(dict_, chk=False):
    if chk:
        if not isinstance(dict_, dict): raise TypeError("dict_ must be a dictionary.")

    return list(dict_.values())


''' will sort dict by values, needs to be tested '''
def sort_dict(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}



'''
Both "objs_to_dict" and "add_objs_to_dict" '''

'''

Boolean Operations

'''

''' seems to have issues... verify later...'''
# will check if value is equal to any from list
# (i.e. (val==el1) or (val==el2) or ...)
def list_or(val,list_):
    bl = True
    for el in list_:
        bl = bl or (val==el)
    return bl



'''

Time Operations

'''

# generate date string to append to file-name
'''
This function was verified on 12/2/2021
'''
def gen_date_ext():
    return replace_chars_with_char(str(datetime.now()),'-: .','_')


'''

Directory Operations

'''

# remove all files in folder path
'''
This function was verified on 12/2/2021
'''
def remove_in_dir(dir_path):
    files = glob.glob(dir_path+'/*')
    for f in files: os.remove(f)

# add a folder to a parent path
'''
This function was verified on 12/2/2021
'''
def make_dir(parent_dir, new_dir):
    os.mkdir(os.path.join(parent_dir, new_dir))
  

# remove all files except select in folder path
# please enter full file name with extension 
'''
This function was verified on 12/2/2021

seems to not remove 'folders', gives "Access Denied" Error.
For now, catching error and continuing (leaving folder there)
'''
def remove_all_except(dir_path,*files):
    all_files = glob.glob(dir_path+'/*')
    files = [dir_path+'\\'+file for file in files]
    for f in all_files: 
        try:
            if not f in files: os.remove(f)
        except PermissionError: continue

'''

Class/Instance (Object) Operations

'''

''' Will return ALL base classes of a class (and not just of the immediate parent) '''
def get_all_bases(cls):
    all_ = []

    def bases(in_):            
        cls_bases = in_.__bases__
        for base_cls in cls_bases:
            all_.append(base_cls)
            bases(base_cls)

    bases(cls)
    return all_
    

'''
Setting multiple attributes of an object at once using key-word arguments
i.e. as:
setattrs(
    obj,
    a=1,
    b=6
    c=9,
    ...
)
'''
def setattrs(_self, **kwargs):
    for k, v in kwargs.items():
        setattr(_self, k, v)


# will return a generator for n deepcopies of any object
'''
This function was verified on 12/14/2021, see test macro: deepcopies_12_14_2021.txt
'''
def deepcopies(obj, n):
    return (deepcopy(obj) for _ in range(n))


# This class strictly wraps any list variable. This is useful for constructs
# that handle lists in two ways. A 'list_wrapper' can be invoked to establish distinction.
class list_wrapper():

    def __init__(self, in_):
        if not isinstance(in_, list): raise TypeError('Input must only be a list.')
        self.__inside = in_
    
    def __iter__(self):
        return (el for el in self.inside)

    @property
    def inside(self):
        return self.__inside

    @inside.setter
    def inside(self, in_):
        if not isinstance(in_, list): raise TypeError('Input must only be a list.')
        self.__inside = in_

# This class strictly wraps any tuple variable. This is useful for constructs
# that handle tuples in two ways. A 'tuple_wrapper' can be invoked to establish distinction.
class tuple_wrapper():

    def __init__(self, in_):
        if not isinstance(in_, tuple): raise TypeError('Input must only be a tuple.')
        self.__inside = in_
    
    def __iter__(self):
        return (el for el in self.inside)

    @property
    def inside(self):
        return self.__inside

    @inside.setter
    def inside(self, in_):
        if not isinstance(in_, tuple): raise TypeError('Input must only be a tuple.')
        self.__inside = in_


# Will set attributes ('attrs') of objects ('objs') to select values ('vals'). 
'''
Not verified... but seems functional...

*** Another feature is to make 'attrs' amd 'vals' distributive (i.e. distribute setting of all 
*** 'attrs' over all 'objs'); will work on this enhancement later...

'''

# def set_attrs(objs, attrs, vals, chk=False):
#     err_msg = ValueError('All inputs that are lists must have same length, greater than one.')

#     def set_it(obj, attr, val):
#         # if at this point, 'in_' won't be a list (already checked for)
#         if isinstance(val, list_wrapper): setattr(obj, attr, val.inside)
#         else: setattr(obj, attr, val)

#     if chk:
#         ''' Checking if inputs are valid types '''
#         if isinstance(objs, list):
#             for obj in objs: illegal_types(obj, *built_ins, input_name='If list, "objs" elements')
#             # if successfully passed, 'objs' is a list of valid class instances (not built-ins)
#         else: illegal_types(objs, *built_ins, input_name='If not list, "objs"')
#         # if successfully passed, 'objs' is a valid class instance (not built-in)

#         if isinstance(attrs, list):
#             for attr in attrs: 
#                 if not isinstance(attr, str): raise TypeError('If list, "attrs" elements must be string objects.')
#             # if successfully passed, 'attrs' is a list of string objects
#         else: 
#             if not isinstance(attrs, str): raise TypeError('If not list, "attrs" must be a string object.')
#         # if successfully passed, 'attrs' is a string object


#         ''' Checking if lists are same sizes, greater than one '''
#         if isinstance(objs, list):
#             x=len(objs)
#             if x<=1: raise err_msg

#             if isinstance(attrs, list):
#                 y=len(attrs)
#                 if y!=x: raise err_msg

#                 if isinstance(vals, list):
#                     z=len(vals)
#                     if z!=x: raise err_msg
#             else:
#                 if isinstance(vals, list):
#                     z=len(vals)
#                     if z!=x: raise err_msg  
#         else:
#             if isinstance(attrs, list):
#                 y=len(attrs)
#                 if y<=1: raise err_msg

#                 if isinstance(vals, list):
#                     z=len(vals)
#                     if z!=y: raise err_msg
#             else:
#                 if isinstance(vals, list):
#                     z=len(vals)
#                     if z<=1: raise err_msg          


#     ''' actual execution'''    
#     if isinstance(objs, list):
#         if isinstance(attrs, list):
#             if isinstance(vals, list):
#                 for obj, attr, val in zip(objs, attrs, vals):
#                     set_it(obj, attr, val)
#             else:
#                 for obj, attr in zip(objs, attrs):
#                     set_it(obj, attr, vals)
#         else:
#             if isinstance(vals, list):
#                 for obj, val in zip(objs, vals):
#                     set_it(obj, attrs, val)
#             else:
#                 for obj in objs:
#                     set_it(obj, attrs, vals)
#     else:
#         if isinstance(attrs, list):
#             if isinstance(vals, list):
#                 for attr, val in zip(attrs, vals):
#                     set_it(objs, attr, val)
#             else:
#                 for attr in attrs:
#                     set_it(objs, attr, vals)
#         else:
#             if isinstance(vals, list):
#                 raise ValueError('"vals" cannot be list when "attrs" and "objs" are both not; \n  \
#                                   cannot set multiples values to single attribute of single   \n \
#                                   object. If intended to pass a single value that is list, use\n \
#                                   a list-wrapper.')
#             else:
#                 set_it(objs, attrs, vals)


# # Will call methods ('mthds') of objects ('objs') to select inputs ('ins') (put 'None' if doesn't take any). 

# '''
# Not verified... but seems functional...

# *** Another feature is to make 'attrs' amd 'vals' distributive (i.e. distribute setting of all 
# *** 'attrs' over all 'objs'); will work on this enhancement later...

# '''

# def do_methods(objs, mthds, ins=None, chk=False):

#     err_msg = ValueError('All inputs that are lists must have same length, greater than one.')

#     def do_action(action_,in_):
#         # if at this point, 'in_' won't be a list (already checked for)
#         if isinstance(in_, tuple): action_(*in_)
#         elif isinstance(in_, tuple_wrapper) or isinstance(in_, list_wrapper): action_(in_.inside)
#         elif in_==None: action_()
#         else: action_(in_)

#     if chk:
#         ''' Checking if inputs are valid types '''
#         if isinstance(objs, list):
#             for obj in objs: illegal_types(obj, *built_ins, input_name='If list, "objs" elements')
#             # if successfully passed, 'objs' is a list of valid class instances (not built-ins)
#         else: illegal_types(objs, *built_ins, input_name='If not list, "objs"')
#         # if successfully passed, 'objs' is a valid class instance (not built-in)

#         if isinstance(mthds, list):
#             for mthd in mthds: 
#                 if not isinstance(mthd, str): raise TypeError('If list, "mthds" elements must be string objects.')
#             # if successfully passed, 'mthds' is a list of string objects
#         else: 
#             if not isinstance(mthds, str): raise TypeError('If not list, "mthds" must be a string object.')
#         # if successfully passed, 'mthds' is a string object


#         ''' Checking if lists are same sizes, greater than one '''
#         if isinstance(objs, list):
#             x=len(objs)
#             if x<=1: raise err_msg

#             if isinstance(mthds, list):
#                 y=len(mthds)
#                 if y!=x: raise err_msg

#                 if isinstance(ins, list):
#                     z=len(ins)
#                     if z!=x: raise err_msg
#             else:
#                 if isinstance(ins, list):
#                     z=len(ins)
#                     if z!=x: raise err_msg  
#         else:
#             if isinstance(mthds, list):
#                 y=len(mthds)
#                 if y<=1: raise err_msg

#                 if isinstance(ins, list):
#                     z=len(ins)
#                     if z!=y: raise err_msg
#             else:
#                 if isinstance(ins, list):
#                     z=len(ins)
#                     if z<=1: raise err_msg          


#     ''' actual execution'''    
#     if isinstance(objs, list):
#         if isinstance(mthds, list):
#             if isinstance(ins, list):
#                 for obj, mthd, in_ in zip(objs, mthds, ins): do_action(getattr(obj, mthd), in_)
#             else:
#                 for obj, mthd in zip(objs, mthds): do_action(getattr(obj, mthd), ins)
#         else:
#             if isinstance(ins, list):
#                 for obj, in_ in zip(objs, ins): do_action(getattr(obj, mthds), in_)
#             else:
#                 for obj in objs: do_action(getattr(obj, mthds), ins)
#     else:
#         if isinstance(mthds, list):
#             if isinstance(ins, list):
#                 for mthd, in_ in zip(mthds, ins): do_action(getattr(objs, mthd), in_)
#             else:
#                 for mthd in mthds: do_action(getattr(objs, mthd), ins)
#         else:
#             if isinstance(ins, list):
#                 raise ValueError('"ins" cannot be list when "mthds" and "objs" are both not;   \n \
#                                   cannot call single method multiple times for different inputs\n \
#                                   on single object. If intended to call method once with "ins" \n \
#                                   as a list input, use a list-wrapper.')

#             else: do_action(getattr(objs, mthds), ins)

'''
class attributes to lists
'''
def attrs_to_list(obj):
    return list(obj.__dict__.values())



'''
**************************************************************

Hyperract Constructs: N-Dimensional Array Construct(s) (i.e. vector 
    (1-D), matrix (2-D), cube (3-D), tesseract (4-D), etc...) 

**************************************************************
'''


'''

Hyperract Input-Checks (Primarily used for module hyperract.py)

'''



def is_array(arr):
    try: len(arr)
    except: return False
    else: return True


def is_array_nums(arr):
    try: 
        if len(arr)==0: raise ValueError('"arr" is empty.')
        else:
            for el in arr:
                if not isinstance(el,tuple(num_types)): return False
            return True
    except: raise TypeError('"arr" must be an array; must possess "len()".')



def is_array_arrays(arr):
    try: 
        len(arr)
        for el in arr:
            try: len(el)
            except: return False
        return True
    except: raise TypeError('"arr" must be an array; possess "len()".')    




def is_hyperray(H):
    try:
        if is_array_nums(H): return True
        elif is_array_arrays(H): 
            for h in H:
                if not is_hyperray(h): return False
            return True
        return False
    except: return False

def are_hyperrays(*H):
    for h in H:
        if not is_hyperray(h): return False
    return True


# By default, can only detect hyperracts with a maximum dimension of 100.
def is_hyperract(H,max_dim=None):
    def cmp_gen(dim):
        def wrap(h,H):
            if dim == 2: return len(h)==len(H[0])
            else:
                cmp_in = len(h)==len(H[0])
                for _ in range(dim-2):
                    h = h[0]
                    H = H[0]
                    cmp_in = (cmp_in and (len(h)==len(H[0])))
                return cmp_in
        return wrap

    def hyper_dec(orig,cmp_in):
        def wrap(H):
            if is_array(H):
                if all([orig(h) for h in H]):
                    if all([cmp_in(h,H) for h in H]): return True
                    else: return False
                else: return False
            else: return False
        return wrap

    def hyper(H):
        if is_array(H):
            if all([(isinstance(h,tuple(num_types)) or h==None) for h in H]): return True
            else: return False
        else: return False

    dim = 1
    tmp = hyper(H)

    if max_dim==None: mx=100
    elif isinstance(max_dim, tuple(int_types)) and max_dim>0: mx = max_dim
    else: raise ValueError('"max_dim", if entered, must be a positive integer')

    while (tmp == False) and (dim <= mx):
        dim += 1
        hyper = hyper_dec(hyper,cmp_gen(dim))
        tmp = hyper(H)

    return tmp

def are_hyperracts(*H):
    for h in H:
        if not is_hyperract(h): return False
    return True



def hyperract_size(H):
    if is_hyperract(H):
        size = []
        tmp = True
        while tmp:
            try:
                size.append(len(H))
                H = H[0]
            except: tmp = False
        return size
    else: raise TypeError('H must be a hyperract.')





'''
    Below are relevant input checks (only ever used for
    this module.)
'''








'''
Notice: Need to look for new error handling methods. Current methods have a lot of
        redundancies.

Notice: 'main' hyper_transpose() (for all n dimensions) producing the correct tranposed hyperract on all 
        permutations as of 5/11/2021. Condensed hyper_transpose() expanded for first six dimensions archived
        5/11/2021. 
'''


def transpose(M):
    if (is_hyperract(M)==True) and (len(hyperract_size(M))==2):
        return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]
    else:
        # Raise Error Later.
        print('Invalid Input. M must be a matrix.')



def hyper_transpose(H,perm):
    # Needs input check for permutations
    def permute(H,perm,dim,*inds):
        vec0 = [[dim-s-1,inds[s]] for s in range(dim)]
        vecP = list(list(permutations(vec0))[perm])
        permN = [vec0[vecP.index(vec0[s])][1] for s in range(dim)]
        for i in range(dim): H = H[permN[i]]
        return H

    def transpose_loop(h,prm,loop,inds,dm,dm_szs):
        if loop==dm-1: 
            return [permute(h,prm,dm,*[*inds,ind]) for ind in range(dm_szs[loop])]
        else: 
            return [transpose_loop(h,prm,loop+1,[*inds,ind],dm,dm_szs) for ind in range(dm_szs[loop])]

    if is_hyperract(H):
        dim = len(hyperract_size(H))
        dim_sizes = [len(H)]
        Htmp = H 
        for i in range(dim-1): 
            Htmp = Htmp[0]
            dim_sizes.append(len(Htmp))
        dim_sizes = list(list(permutations(dim_sizes))[perm])
        del Htmp

        return transpose_loop(H,perm,0,[],dim,dim_sizes)
    else:
        raise TypeError('H must be a hyperract.')




'''
Notice: Need to look for new error handling methods. Current methods have a lot of
        redundancies and hidden uncertainties.

Notice: 'current' hyper_statisize() applies to all hyperracts with dimensions of 3 or greater. Condensed version
        valid for up to 6 dimensions archived 5/10/2021
'''




stat_funcs = {
    None: np.mean,
    'avg': np.mean,
    'std': np.std,
    'sum': np.sum,
    'max': np.max,
    'min': np.min
}



def statisize(M,func=None):
    if func in stat_funcs:
        if is_hyperract(M) and (len(hyperract_size(M)) == 2): 
            return [stat_funcs[func](col) for col in transpose(M)]
        else: raise TypeError('M must be a matrix.')
    else: raise ValueError('"func" must be from "stat_funcs".')


def hyper_statisize(H,depth,func=None):

    def statisize_dec(orig,dim):
        def wrap(H,depth,func):
            if depth == dim-1:
                for i in range(dim-2): H = concatenate(*H)
                return statisize(H,func)
            else: return [orig(h,depth,func) for h in H]
        return wrap

    def statisize_loop(H,depth,func): return statisize(H,func)

    try:
        dim = len(hyperract_size(H))
        if dim>=3 and isinstance(depth,tuple(int_types)) and depth>=0 and depth<dim:
            if depth==0: return H
            else:
                for i in range(dim-2): statisize_loop = statisize_dec(statisize_loop,3+i)
                return statisize_loop(H,depth,func)
        else: raise ValueError('H must have 2 or more dimensions and "depth" must be a positive integer less than the number of dimensions.')
    except TypeError: raise TypeError('H must be a hyperract.')






'''
    Below is the hyperract class itself and its associated
    configuration inputs.

'''


def pars_setter(func,*pars):
    def wrap():
        try: return func(*pars)
        except:
            # This may compete with the functions in-built error-handling methods, more on this...
            raise ValueError('Invalid number of pars or invalid par types.')
    return wrap



pop_funcs = {
    None: np.mean,
    'rrand': np.mean,
    'std': np.std,
    'sum': np.sum,
    'max': np.max,
    'min': np.min,
    'rrand':rrand,
    'urrand':pars_setter(rrand,0,10)
}



class hyperract():
        
    def level_loop(self,method=None):
        def wrap(lvl,vals):
            if lvl==len(vals)-1:
                # list of None currently recognized as hyperract, but more on what to do with this later...
                if method==None: return [None for ind in range(len(vals[lvl]))]
                else:
                    try: return [method() for ind in range(len(vals[lvl]))]
                    except:
                        # Later restrict functions to only be from pop_funcs dictionary, more on this...
                        raise NameError('Invalid function input. Must be from "pop_funcs" dictionary.')
            else: return [wrap(lvl+1,vals) for ind in range(len(vals[lvl]))]
        return wrap


    # if type is non-numerical, meaning it can hold strings, then disable all statisize methods (i.e. mean, std, etc..), more on this...
    def __init__(self,keys,type=None):
        
        if isinstance(keys,dict):
            if all([isinstance(val,list) for val in keys.values()]):
                self.keys = keys
                self.vals = list(keys.values())
                self.hyperract = self.level_loop()(0,self.vals)
            else: raise ValueError('Dictionary must contain lists as values.')
        else: raise TypeError('Input must be a dictionary.') 

    def populate(self,method):
        self.hyperract = self.level_loop(method=method)(0,self.vals)


    # plan to make this method more intuitive (i.e. inputting the new desired sequence instead of the permutation number itself
    # which is not intuitve...)
    def transpose(self,perm):
        try:
            self.hyperract = hyper_transpose(self.hyperract,perm)
        except:
            raise ValueError('Please enter a valid permutation.')


'''
**************************************************************

CSV-parsers

**************************************************************
'''



# Needs Input Check
def str_to_flt_list(arr):
    return [float(el) for el in arr]

# Needs Input Check
def flt_to_int_list(arr):
    return [int(el) for el in arr]

# Needs Input Check
def row_to_csv(row,file,new=False):
    if new: w = 'w' 
    else: w = 'a'
    # Needs handling for list of strings
    with open(file,w,newline='') as f: csv.writer(f).writerow(row)

# Needs Input Check
def csv_to_list(file):
    with open(file) as f: return list(csv.reader(f, delimiter=','))




def hyper_to_csv(H,file):
    def write_loop(H,lvl):
        if lvl==1:
            row_to_csv(H,file)
        else:
            for h in H: write_loop(h,lvl-1)
    try:
        szs = hyperract_size(H)
        row_to_csv(['Hyperract Tag'],file,new=True)
        row_to_csv(szs,file)
        write_loop(H,len(szs))
    except TypeError:
        raise TypeError('H must be a hyperract')





def csv_to_hyper(file):

    def indexer(inds,szs):
        sz = 1
        idx = 0
        n = len(szs)-2
        for i in range(n+1):
            idx += sz*inds[i]
            if i<=n: sz *= szs[n-i]
        return idx

    def extract_loop(arr,inds,lvl,szs):
        if lvl==len(szs)-2:
            return [arr[indexer([ind,*inds],szs)] for ind in range(szs[lvl])]
        else:
            return [extract_loop(arr,[ind,*inds],lvl+1,szs) for ind in range(szs[lvl])]

    arr = csv_to_list(file)
    if arr[0] == ['Hyperract Tag']:
        try: 
            arr.pop(0)
            arr = [str_to_flt_list(row) for row in arr]
            szs = flt_to_int_list(arr[0])
            arr.pop(0)
            return extract_loop(arr,[],0,szs)
        except:
            raise TypeError('csv file has tag but does not contain a hyperract.')
    else:
        raise TypeError('csv file must have "Hyperract Tag" on first row.')



