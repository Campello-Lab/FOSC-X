import numpy as np

def B3_F_Measure(cluster_tree,labels,N_nodes):

    Fb3 = []

    for node in range(N_nodes):

        obs = cluster_tree.get_node_indices(node)
        node_labels = labels[obs]

        # labeled objects only
        labeled_mask = node_labels >= 0
        labeled_node_labels = node_labels[labeled_mask]

        denomP = len(labeled_node_labels)

        if denomP == 0:
            Fb3.append(0.0)
            continue

        numer = []
        denomR = []
        weight = []

        for i in np.unique(labeled_node_labels):

            n_i = np.sum(labeled_node_labels == i)

            numer.append(n_i)
            denomR.append(np.sum(labels == i))
            weight.append(n_i)

        numer = np.array(numer, dtype=float)
        denomR = np.array(denomR, dtype=float)
        weight = np.array(weight, dtype=float)

        B3Precision = numer / denomP
        B3Recall = numer / denomR

        F = 2 * B3Precision * B3Recall / (B3Precision + B3Recall)

        Fb3temp = np.sum(weight * F)

        Fb3.append(Fb3temp)

    Fb3 = np.array(Fb3,dtype=float)
    Nl = sum(labels>-1)
    if Nl>0:
        Fb3 /= sum(labels>-1)
    else:
        Fb3 = 0

    return(Fb3)