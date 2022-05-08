import numpy as np
import matplotlib.pyplot as plt
import ast
import os

def read_file(filename):
    '''
    Read the file and return the data.
    '''
    with open(filename, "r") as f:
        file_str = f.read()
        data = ast.literal_eval(file_str)
        n_iter = data[0][0]
        grid = data[1]
        grid = np.array(grid)
    return (n_iter, grid)


def generate_plots(filename):
    '''
    Plot and save the data.
    '''
    n_iter, data = read_file(filename)
    plt.clf()
    plt.imshow(data, cmap=plt.cm.get_cmap('binary_r'))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(orientation='vertical')
    plt.title('Conjugate Gradient Plot')

    plt.savefig('.' + filename.split('.')[1] + '.png')
    
    plt.close()



if __name__ == '__main__':
    generate_plots('./output/output.txt')