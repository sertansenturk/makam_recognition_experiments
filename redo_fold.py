import sys
import os
import json
from dlfm_code.tester import test

def main():
    idx = int(sys.argv[1])

    ##
    errors = json.load(open('res_err.json', 'r'))
    error_params = [err[1] for err in errors]
    
    print str(idx) + ': ' + errors[idx][0]
    test(*errors[idx][1], overwrite=True)

if __name__ == "__main__":
    main()