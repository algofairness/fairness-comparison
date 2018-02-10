#!/bin/bash

PAPER_PATH=$1

if [ "$PAPER_PATH" == "" ]; then
    echo "Usage: $0 path-to-paper-directory"
    exit 1
fi

if [ ! -e ${PAPER_PATH}/paper.tex ]; then
    echo "Error: ${PAPER_PATH}/paper.tex not found. Are you sure this is the correct path?"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cp * ${PAPER_PATH}/figs/analysis
rm ${PAPER_PATH}/figs/analysis/copy-to-paper.sh


