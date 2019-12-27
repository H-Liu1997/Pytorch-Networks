# --------------------------------------------------------------------------- #
# Sample gnn, this is keras version
# implement by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #

import numpy as np 
import networkx as nx
from keras.activations import relu
from networkx import to_numpy_matrix

if __name__ == "__main__":

    A = np.matrix([
        [0,1,0,0],
        [0,0,1,1],
        [0,1,0,0],
        [1,0,1,0]], 
        dtype = float)

    X = np.matrix([
       [i,-i] for i in range(A.shape[0])
       ], dtype = float)

    print(X)

    Y = A * X #4*2 matrix 
    print(Y)

    #for add self info
    I =  np.matrix(np.eye(A.shape[0]))
    A_hat = A + I
    Y_hat = A_hat * X
    print (Y_hat)

    # calculate degree matirx for in
    D = np.array(np.sum(A,axis = 0))[0] # ?
    print ("D1",D)
    D = np.matrix(np.diag(D))
    print (D)

    # new formula
    Y_d =  D ** -1 * A * X
    print (Y_d)

    D_hat = np.array(np.sum(A_hat,axis = 0))[0]
    D_hat = np.matrix(np.diag(D_hat))
    W = np.matrix([
        [1,-1],
        [-1,1]
    ])
    Y_d_hat = D_hat ** -1 * A_hat * X * W # 4*4 * 4*4 * 4*2 * 2*2 = 4*2
    print(Y_d_hat)
    Y_final = relu(Y_d_hat)
    print(Y_final[1])
    

    zkc = nx.karate_club_graph()
    order = sorted(list(zkc.nodes()))
    A_zkc = to_numpy_matrix(zkc,nodelist=order)
    print(A_zkc)
    I_zkc = np.eye(zkc.number_of_nodes())
    print(I_zkc)
    A_zkc_hat = A_zkc + I_zkc
    print(A_zkc_hat)
    D_zkc_hat = np.array(np.sum(A_zkc_hat,axis = 0))[0]
    D_zkc_hat = np.matrix(np.diag(D_zkc_hat))
    W_1 = np.random.normal(
        loc = 0, scale = 1, size = (zkc.number_of_nodes(),4)
    )
    W_2 = np.random.normal(
        loc = 0, scale = 1, size = (W_1.shape[1], 2)
    )
    print(W_2)

    def gcn_layer(A_zkc_hat,D_zkc_hat, X ,W):
        return relu(np.dot(np.dot(np.dot(D_zkc_hat**-1, A_zkc_hat), X), W))

    H_1 = gcn_layer(A_zkc_hat, D_zkc_hat, I_zkc, W_1)
    print(H_1)
    
    H_1_mat = np.matrix(H_1)
    print(H_1_mat.shape)
    H2 = np.dot(D_zkc_hat**-1, A_zkc_hat)
    print(H2.shape)
    h3 = np.dot(H2,H_1)
    print(h3.shape)
    H_2 = gcn_layer(A_zkc_hat, D_zkc_hat, H_1, W_2)

    output = H_2
    print(H_2)

    feature_representations = {
        node: np.array(output)[node]
        for node in zkc.nodes()
    }   


