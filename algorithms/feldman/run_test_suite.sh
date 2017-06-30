# Run this file with bash to run all Python files in this directory and in sub-directories.
# Note: excludes "__init__.py" files, "Abstract" files, and the "main.py" file.

# To implement a test, use the following format in the file-to-test:
#
# if __name__=="__main__": test()
# def test():
#   ...
#

# Add the current directory to the PYTHONPATH so imports start at the project root.
export PYTHONPATH="${PYTHONPATH}:`pwd`"

echo "#########################################################################"
echo "### Running all *.py files now. #########################################"
echo "### No tests should be False nor should there be Traceback exceptions. ##"
echo "#########################################################################"

# Loop largely based on: http://stackoverflow.com/questions/15065010/how-to-perform-a-for-each-file-loop-by-using-find-in-shell-bash
find . -type f -iname "*.py" -print0 | while IFS= read -r -d $'\0' line; do
  if [[ ! $line =~ .*__init__.py ]]; then
    if [[ ! $line =~ .*Abstract.+.py ]]; then
      if [[ ! $line =~ ./main.py ]]; then
        if [[ ! $line =~ ./histogram_maker.py ]]; then
          echo "________________________________"
          echo "Running tests for: $line"
          python "$line" | grep --color -E '^|False$' # Highlight "False" tests.
        fi
      fi
    fi
  fi
done

