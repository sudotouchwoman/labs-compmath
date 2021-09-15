import numpy as np

def cubic_spline_coeff(x_nodes, y_nodes):
    '''
        Compute and return `(N-1) x 3` matrix of cubic spline coefficients

        + `x_nodes`: list or 1darray
            array of base nodes' x axis
        
        + `y_nodes`: list or 1darray
            array of base nodes' y axis
    '''
    
    # create intermediate vectors for neighbouring nodes' x deltas and copy y values to a (same as a_i) (just to be more clear)
    h = np.array([ (x_nodes[i+1] - x_nodes[i]) for i, _ in enumerate(x_nodes[:-1]) ])
    a = np.array(y_nodes)

    # create A matrix (required to compute the 1st vector of spline parameters, c_i)
    # afterwards we will iteratively compute b_i and d_i as a_i is already here for us being set of Y axis values of nodes
    def build_A():
        
        n = len(x_nodes) 
        # define generators for diagonal sequences
        # (seemed convenient to me)
        def main_diag(n: int):
            yield 1
            for i in range(1, n - 1):
                h_1 = h[i-1]
                h_2 = h[i]
                yield 2*(h_1 + h_2)
            yield 1

        def up_diag(n: int):
            yield 0
            for i in range(1, n - 1):
                yield h[i]

        def down_diag(n: int):
            for i in range(n - 2):
                yield h[i]
            yield 0
        
        # form A matrix by adding 3 diagonals together
        A = np.diag([item for item in main_diag(n)],k=0) \
            + np.diag([item for item in up_diag(n)], k=1) \
            + np.diag([item for item in down_diag(n)], k=-1)
        
        return A

    def build_b():
        n = len(x_nodes)

        # define generator for b vector 
        # (do not mix with b_i, vector of spline coeffs, former is a part of matrix equation A @ c_i = b)
        def fill_b(n: int):
            yield 0
            v_item = lambda idx:\
                (3/h[idx])*(a[idx+1] - a[idx]) - (3/h[idx-1])*(a[idx] - a[idx-1])
            for i in range(1, n - 1):
                yield v_item(i)
            yield 0

        return np.array([item for item in fill_b(n)])

    # form matrix A and vector b from matrix equation for c_i
    # as A @ c_i = b -> c_i = inv(A) @ b
    A = build_A()
    b = build_b()

    # solve matrix equation naively by using inverse matrix
    # to compute c_i coeff vector
    c = np.linalg.inv(A) @ b

    # compute d_i and b_i coeffs using iterative formulas
    b_i = lambda i:\
        (1/h[i])*(a[i+1] - a[i]) - (h[i]/3)*(c[i+1] + 2*c[i])

    d_i = lambda i:\
        (c[i+1] - c[i])/(3*h[i])

    b = np.array([b_i(i) for i, _ in enumerate(x_nodes[:-1])])
    d = np.array([d_i(i) for i, _ in enumerate(x_nodes[:-1])])

    # uncomment below to check intermediate results
    # print(f'A (matrix) = {A}')
    # print(f'A = {a}')
    # print(f'C = {c}')
    # print(f'D = {d}')
    # print(f'B = {b}')

    # concatenate b_i, c_i, d_i vectors 
    # (also chop off last c_i item as c_i.shape() is (n,) and we want to get (n-1, 3) coeff matrix)
    return np.c_[b, c[:-1], d]

if __name__ == '__main__':
    # simple check (for debug purposes) of method performance
    # note that json should be in this directory (was moved away)
    import json
    with open('nodes.json','r') as infile:
        NODES = json.loads(infile.read())
        print(f'NODES:\n{NODES}')
    
    coeffs = cubic_spline_coeff( [node['X'] for node in NODES], [node['H'] for node in NODES] )
    print(f'COEFFS:\n{coeffs}')