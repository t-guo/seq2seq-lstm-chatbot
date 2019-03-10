import os, sys

# get base directory as from cmc folder level, e.g. basedir/app/...
basedir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# append modules from basedir so they can be used in flask framework
sys.path.append(basedir)