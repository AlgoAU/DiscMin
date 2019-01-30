import numpy as np
import discpy as disc

def sparsify(a,x,norm=np.inf,local_search=0.3):
    x = x.copy()
    n = a.shape[1]
    abs_x = np.absolute(x)
    sorted_indices = np.argsort(-abs_x)
    is_zero = (x==0)
    not_zero = ~is_zero
    num_non_zero = not_zero.sum()
    largest_index = num_non_zero-1
    mid_val = abs_x[int(largest_index/2)]
    smallest_index = int(largest_index/2)
    while smallest_index>0:
        if abs_x[smallest_index-1] > mid_val*1.5:
            break
        smallest_index = smallest_index-1

    interval_length = largest_index-smallest_index+1
    #we will half number of non-zeroes in interval smallest_index to largest_index
    indices = sorted_indices[smallest_index : largest_index+1]
    sub_x = x[indices]
    sub_a = a[:,indices]
    scaled_a = np.multiply(sub_a,sub_x)
        
    y = disc.discrepancy_minimize(scaled_a,norm,False,local_search)
    num_ones = (y==1).sum()
    t = -1
    if num_ones<=len(indices)/2:
        t=1
    for i in range(len(indices)):
        if y[i]==t:
            sub_x[i] = sub_x[i]*2
        else:
            sub_x[i] = 0
    x[indices] = sub_x
    return x


def sparsify_alternatives(a,x,norm=np.inf,local_search=0.3):
    x = x.copy()
    n = a.shape[1]

    abs_x = np.absolute(x)
    sorted_indices = np.argsort(-abs_x)
    is_zero = (x==0)
    not_zero = ~is_zero
    num_non_zero = not_zero.sum()
    if num_non_zero <= 1:
        return
    largest_index = num_non_zero-1
    mid_val = abs_x[int(largest_index/2)]
    smallest_index = int(largest_index/2)
    while smallest_index>0:
        if abs_x[smallest_index-1] > mid_val*1.5:
            break
        smallest_index = smallest_index-1

    interval_length = largest_index-smallest_index+1
    #we will half number of non-zeroes in interval smallest_index to largest_index
    indices = sorted_indices[smallest_index : largest_index+1]
    sub_x = x[indices]
    sub_a = a[:,indices]
    scaled_a = np.multiply(sub_a,sub_x)
        
    y = disc.discrepancy_minimize(scaled_a,norm,True,local_search)
    sub_x_alt = sub_x.copy()
    for i in range(len(indices)):
        if y[i]==1:
            sub_x[i] = sub_x[i]*2
            sub_x_alt[i] = 0
        else: 
            sub_x_alt[i] = sub_x[i]*2
            sub_x[i] = 0
    x_alt = x.copy()
    x[indices] = sub_x
    x_alt[indices] = sub_x_alt
    return [x,x_alt]

def sparsify_unbiased(a,x,norm=np.inf,local_search=0.3):
    xs = sparsify_alternatives(a,x,norm,local_search)
    if np.random.rand > 0.5: return xs[0]
    else: return xs[1]
