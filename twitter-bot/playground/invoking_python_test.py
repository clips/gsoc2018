import sys, json
import numpy as np

def main ():
    _input = sys.stdin.readlines()
    # print(_input)
    # input_array = None
    # for line in _input:
    #     input_array = np.concatenate(input_array, np.array(json.loads(line)))
    # summed_input = np.sum(input_array)

    print('Input 0: ', _input[0])
    print('Input 1: ', _input[1])

    input_array = np.array(json.loads(_input[0]))
    summed_input = np.sum(input_array)
    
    # Standard stream is routed back to Node here so we can just use print()
    print(summed_input)
    sys.stdout.flush()


if __name__ == '__main__':
    main()    