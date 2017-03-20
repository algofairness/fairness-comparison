#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert ARFF format to space/tab/comma separated format

SYNOPSIS::

    SCRIPT [options] [<INPUT> [<OUTPUT>]]

Options
-------
-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
-d <DL>, --delimiter <DL>
    column delimiter string. 't' specifies tab character.(default " ")
-n <NAN>, --nan <NAN>
    specifies Not Available Number string (default "nan")
-m <MODE>, --mode <MODE>
    decoding mode for nominal values. 1:integer, 2:string, others:binary
    (default binary)
--version
    show version
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2011/02/06"
__version__ = "2.0.0"
__copyright__ = "Copyright (c) 2011 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"
__docformat__ = "restructuredtext en"

#==============================================================================
# Imports
#==============================================================================

import sys
import argparse
from scipy.io.arff import loadarff

#==============================================================================
# Public symbols
#==============================================================================

__all__ = []

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

#==============================================================================
# Functions
#==============================================================================

def convert_data_with_string(opt, m, d):
    """
    Convert one line. nominal values are encoded with strings.

    Parameters
    opt : options
        command line options
    m : scipy.io.arff.MetaData
        definition of attributes
    d : ary
        one data record

    Returns
    -------
    data_row : list
        list of strings of converted data
    """

    l = []

    # for each attribute
    for i in xrange(len(d)):

        if d[i] == '?':
            l.append(opt.nan)
        else:
            l.append(str(d[i]))

    return l

def convert_data_with_integer(opt, m, d):
    """
    Write one line. nominal values are encoded with integer.

    Parameters
    ----------
    opt : options
        command line options
    m : scipy.io.arff.MetaData
        definition of attributes
    d : ary
        one data record

    Returns
    -------
    data_row : list
        list of strings of converted data
    """

    l = []

    # for each attribute
    for i, a in enumerate(m):

        # NaN value
        if d[i] == '?':
            l.append(opt.nan)

        # numeric value
        elif m[a][0] == 'numeric':
            l.append(str(d[i]))

        # discrete value
        else:
            l.append(str(m[a][1].index(d[i])))

    return l

def convert_data_with_binary(opt, m, d):
    """
    Write one line. nominal values are encoded with binary. Nominal values are
    encoded single 0/1 if the size of domain is 2. When the size of domain is
    larger then 2, i-th value is encoded with list of 0/1's where i-th value
    is 1 and other values are 0.

    Parameters
    ----------
    opt : options
        command line options
    m : scipy.io.arff.MetaData
        definition of attributes
    d : ary
        one data record

    Returns
    -------
    data_row: list
        list of strings of converted data
    """

    l = []

    # for each attribute
    for i, a in enumerate(m):

        # NaN value
        if d[i] == '?':
            if m[a][0] == 'numeric' or len(m[a][1]) <= 2:
                l.append(opt.nan)
            else:
                l.extend(['0'] * len(m[a][1]))

        # numeric value
        elif m[a][0] == 'numeric':
            l.append(str(d[i]))

        # binary discrite value
        elif len(m[a][1]) <= 2:
            l.append(str(m[a][1].index(d[i])))

        # non-binary discrite value
        else:
            v = ['0'] * len(m[a][1])
            v[m[a][1].index(d[i])] = '1'
            l.extend(v)

    return l

#==============================================================================
# Main routine
#==============================================================================

def main(opt):
    """ Main routine that exits with status code 0
    """

    ### main process

    # read arff file
    (data, meta) = loadarff(opt.infile)

    ln = 1

    # process data
    try:
        # string encode mode
        if opt.mode == 2:
            for line in data:
                opt.outfile.write(opt.dl.join(
                    convert_data_with_string(opt, meta, line)) + "\n")

                ln += 1

        # integer encode mode
        elif opt.mode == 1:
            for line in data:
                opt.outfile.write(opt.dl.join(
                    convert_data_with_integer(opt, meta, line)) + "\n")
                ln += 1

        # binary encode mode
        else:
            for line in data:
                opt.outfile.write(opt.dl.join(
                    convert_data_with_binary(opt, meta, line)) + "\n")
                ln += 1

    except ValueError:
        sys.exit("data error found in line " + str(ln) + "\n")

    ### post process

    # close file
    if opt.infile is not sys.stdin:
        opt.infile.close()

    if opt.outfile is not sys.stdout:
        opt.outfile.close()

    sys.exit(0)

### Check if this is call as command script
if __name__ == '__main__':
    ### set script name
    script_name = sys.argv[0].split('/')[-1]

    ### command-line option parsing
    ap = argparse.ArgumentParser(
        description='pydoc is useful for learning the details.')

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)

    # basic file i/o
    ap.add_argument('-i', '--in', dest='infile',
                    default=None, type=argparse.FileType('r'))
    ap.add_argument('infilep', nargs='?', metavar='INFILE',
                    default=sys.stdin, type=argparse.FileType('r'))
    ap.add_argument('-o', '--out', dest='outfile',
                    default=None, type=argparse.FileType('w'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('w'))

    # script specific options
    ap.add_argument('-d', '--delimiter', dest='dl', default=' ')
    ap.add_argument('-n', '--nan', dest='nan', default='nan')
    ap.add_argument('-m', '--mode', dest='mode', type=int, default=0)

    # parsing
    opt = ap.parse_args()

    ### post-processing for command-line options
    # basic file i/o
    if opt.infile is None:
        opt.infile = opt.infilep
    del vars(opt)['infilep']
    if opt.outfile is None:
        opt.outfile = opt.outfilep
    del vars(opt)['outfilep']

    # set meta-data of script and machine
    opt.script_name = script_name
    opt.script_version = __version__

    # the specified delimiter is TAB?
    if opt.dl == 't' or opt.dl == 'T':
        opt.dl = '\t'

    ### call main routine
    main(opt)
