#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate learning and test files for cross-validation

SYNOPSIS::

    SCRIPT [options] [<INPUT> [<OUTPUT>]]

Description
-----------
Divide input data into specified number of blocks. One block is used for test
and the rest blocks are used for learning. These divisions are repeated for
each block.

Options
-------
-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
-e <EXT>, --extension <EXT>
    File extension of otuput files (default "data")
-f <FOLD>, --fold <FOLD>
    The number of blocks (default 5)
-c <CHEAD>, --copyhead <CHEAD>
    The number of lines that are copied to each output file (default 0)
--version
    show version
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2010/04/25"
__version__ = "2.0.0"
__copyright__ = "Copyright (c) 2010 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"
__docformat__ = "restructuredtext en"

#==============================================================================
# Imports
#==============================================================================

import sys
import argparse
import math

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

#==============================================================================
# Main routine
#==============================================================================

def main(opt):
    """ Main routine that exits with status code 0
    """

    ### main process

    # nos of folds
    nf = opt.fold
    if nf <= 1:
        raise SystemExit("illegal number of folds")
    ff = "%0" + str(int(math.ceil(math.log(float(nf), 10.0)))) + "d"

    # copy head
    chead = []
    for i in xrange(opt.chead):
        chead.append(opt.infile.readline().strip('\r\n'))

    # read body
    d = []
    for line in opt.infile.readlines():
        d.append(line.strip('\r\n'))

    # close input file
    if opt.infile is not sys.stdin:
        opt.infile.close()

    # generate data
    for f in xrange(nf):
        lfname = opt.outfile + "@" + (ff % f) + "l." + opt.ext
        tfname = opt.outfile + "@" + (ff % f) + "t." + opt.ext

        # open files
        lfile = open(lfname, "w")
        tfile = open(tfname, "w")

        # output header
        for line in chead:
            lfile.write(line + "\n")
            tfile.write(line + "\n")

        # output files
        for i in xrange(len(d)):
            if f == i % nf:
                tfile.write(d[i] + "\n")
            else:
                lfile.write(d[i] + "\n")

        # close output files
        lfile.close()
        tfile.close()

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
    ap.add_argument('-o', '--out', dest='outfile', default=None, type=str)
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default='output', type=str)

    # script specific options
    ap.add_argument('-e', '--ext', default='data')
    ap.add_argument('-f', '--fold', type=int, default=5)
    ap.add_argument('-c', '--copyhead', dest='chead', type=int, default=0)

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

    ### call main routine
    main(opt)
