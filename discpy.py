import scipy.linalg as lin
import numpy as np
import itertools
import time

#Assume the first r rows of a forms an orthonormal basis
#Returns the projection of x onto the orthogonal complement of the rows of a
#scaled to unit length
def make_orthogonal(a,r,x):
    if np.linalg.norm(x)<1e-8:
        return None
    if r==0:
        return x/np.linalg.norm(x)
    u = x
    for j in range(0,int(r/100)):
        u = u - a[range(j*100,(j+1)*100)].T @ (a[range(j*100,(j+1)*100)] @ u)
    for j in range(int(r/100)*100,r):
        u = u - np.inner(u,a[j]) * a[j]
    norm = np.linalg.norm(u)
    if norm < np.linalg.norm(x)/100:
        return None
    u = u/norm
    return u

def round_coloring(a,x,norm = np.inf, balanced=False):
    start = time.time()
    n = a.shape[1]
    abs = np.absolute(x)
    live = (abs < 1.0-1e-4)
    #first sample each entry
    samples = np.random.random_sample(n)
    signs = np.ones(n)
    signs[samples < (1.0-abs)/2.0] = -1
    flipped = np.multiply(x,signs)
    y = np.sign(flipped)
    y[y==0]=1
    #then try all assignments to the coordinates that were live
    num_live = sum(live)
    if num_live<=10:
        live_indices = [i for i, x in enumerate(live) if x]
        ay = a@y
        sub_a = a[:,live_indices]
        sub_y = y[live_indices]
        sub_ay = sub_a @ sub_y
        a_outside = ay - sub_ay
        best_sub_y = sub_y
        best_norm = np.linalg.norm(ay,ord=norm)
        sign_flips = np.ones(num_live)
        while True:
            at = 0
            while at<num_live and sign_flips[at]==-1:
                sign_flips[at]=1
                at = at+1
            if at==num_live:
                break
            sign_flips[at]=-1
            new_sub_y = np.multiply(sub_y,sign_flips)
            new_sub_ay = sub_a @ new_sub_y
            new_ay = a_outside + new_sub_ay
            new_norm = np.linalg.norm(new_ay,ord=norm)
            if new_norm < best_norm:
                best_norm = new_norm
                best_sub_y = new_sub_y
        y[live_indices] = best_sub_y
    #balance the vector in case we need to
    if balanced:
        while True:
            toFlip = 1
            if sum(y==1) < n/2:
                toFlip = -1
            ofToFlip = sum(y==toFlip)
            needToFlip = int(ofToFlip-n/2)
            if needToFlip==0:
                break
            listOfToFlip = [i for i, x in enumerate(y) if x==toFlip]
            ay = a @ y
            best_norm = 1e27
            #try all single flips
            best_to_flip = 0
            for i in listOfToFlip:
                col = a[:,i]
                new_ay = ay - (2*col*y[i])
                new_norm = np.linalg.norm(new_ay,ord=norm)
                if new_norm < best_norm:
                    best_norm = new_norm
                    best_to_flip = i
            y[best_to_flip] = -y[best_to_flip]
    return y

def to_square(a,x,balanced=False):
    n = a.shape[1]
    m = a.shape[0]
    if n<=m:
        return x
    live = n
    orth = np.zeros((n,n))
    is_live = np.ones(n)
    num_orth = 0
    for i in range(0,m):
        p = make_orthogonal(orth,num_orth,a[i])
        if not p is None:
            orth[num_orth] = p
            num_orth = num_orth + 1
    if balanced:
        all_ones = np.ones(n)
        p = make_orthogonal(orth,num_orth,all_ones)
        if np.linalg.norm(p) > 0.01:
            orth[num_orth] = p
            num_orth = num_orth + 1
    while num_orth < n and sum(is_live) > 8:
        #sample a random vector
        g = np.random.randn(n)
        #make it orthogonal to all in orth
        gamma = make_orthogonal(orth,num_orth,g)
        if gamma is None:
            break
        #set all that are not live to 0 for numerical stability
        for i in range(0,n):
            if not is_live[i]:
                gamma[i] = 0
        #find coord that freezes first
        delta = 1e30
        for i in range(0,n):
            if is_live[i] and abs(gamma[i]) > 1e-7:
                dist = 0
                if gamma[i] * x[i] >= 0:
                    dist = (1-abs(x[i]))/abs(gamma[i])
                else:
                    dist = (1+abs(x[i]))/abs(gamma[i])
                if dist < delta:
                    delta = dist
        if delta > 1e25:
            break
        x = x + delta * gamma
        for i in range(0,n):
            if is_live[i] and abs(x[i]) >= 1-1e-6:
                is_live[i] = 0
                e = np.zeros(n)
                e[i] = 1
                p = make_orthogonal(orth,num_orth,e)
                if not p is None:
                    orth[num_orth] = p
                    num_orth = num_orth+1
                    if num_orth>=n:
                        break
    return x

def partial_color(a,x,norm=np.inf,balanced=False):
    if norm==np.inf:
        return partial_infty_color(a,x,balanced)
    if norm==2:
        return partial_l2_color(a,x,balanced)

def partial_l2_color(a, x, balanced=False):
    n = a.shape[1]
    m = a.shape[0]
    delta=1e-5

    #count number of live coordinates
    abs_x = np.absolute(x)
    is_live = (abs_x < 1.0-delta)
    initial_is_live = is_live.copy()
    live = is_live.sum()
    if live<8: 
        return True
    initial_live=live
    #extract the live coordinates and columns
    y = x[initial_is_live]
    b = a[:,initial_is_live]
    #compute eigenvectors of bTb
    bTb = b.T @ b
    w, v = lin.eigh(bTb)
    num_iters = 0
    was_live = np.ones(initial_live,dtype=bool)
    num_frozen = 0
    orth = np.zeros((initial_live,initial_live))
    num_orth = 0
    to_add = v[:,range(initial_live-int(initial_live/2), initial_live)].T
    orth[range(0,to_add.shape[0])] = to_add
    num_orth = to_add.shape[0]
    if balanced:
        ones = np.ones(initial_live)
        ones = make_orthogonal(orth,num_orth,ones)
        if not ones is None:
            orth[num_orth] = ones
            num_orth = num_orth+1
    while int((live*3/2)) > initial_live:
        #will have at most 1/3 * initial_live that are frozen
        abs_y = np.absolute(y)
        #start by freezing new frozen coordinates
        is_live = (abs_y < 1.0-delta)
        diff = is_live!=was_live
        num_diff = diff.sum()
        if num_diff>0:
            for i in range(0,initial_live):
                if diff[i]:
                    e = np.zeros(initial_live)
                    e[i] = 1
                    p = make_orthogonal(orth,num_orth,e)
                    if not p is None:
                        orth[num_orth] = p
                        num_orth = num_orth + 1
                        if num_orth>=initial_live:
                            break
                    num_frozen = num_frozen + 1
            was_live = is_live
        if num_orth>=initial_live:
            break
        num_iters = num_iters+1
        ax = a @ x
        #sample a random vector
        g = np.random.randn(initial_live)
        #make it orthogonal to all in orth
        gamma = make_orthogonal(orth,num_orth,g)
        if gamma is None:
            break
        #set coords to zero where not live, due to numerical stability
        for i in range(0,initial_live):
            if not is_live[i]:
                gamma[i] = 0
        if np.linalg.norm(gamma,ord=np.inf)==0:
            break
        if np.inner(ax,b @ gamma) > 0:
            gamma = -gamma
        coord_mult = np.multiply(gamma,y)
        val = np.where(coord_mult < 0, (1+abs(y))/(1e-27+abs(gamma)), (1-abs(y))/(1e-27 + abs(gamma)))
        val = np.where(is_live, val, 1e27)
        z = min(val)
        y = y + z*gamma
        x[initial_is_live] = y
        new_abs_y = np.absolute(y)
        is_live = (new_abs_y < 1.0-delta)
        live = is_live.sum()
    return False

def partial_infty_color(a, x, balanced=False):
    n = a.shape[1]
    m = a.shape[0]
    delta=1e-5

    #count number of live coordinates
    abs_x = np.absolute(x)
    is_live = (abs_x < 1.0-delta)
    initial_is_live = is_live.copy()
    live = is_live.sum()
    if live<8: 
        return True
    initial_live=live
    #extract the live coordinates and columns
    y = x[initial_is_live]
    b = a[:,initial_is_live]
    num_iters = 0
    was_live = np.ones(initial_live,dtype=bool)
    was_small = np.ones(m,dtype=bool)
    orth = np.zeros((initial_live,initial_live))
    num_orth = 0
    if balanced:
        ones = np.ones(initial_live)
        ones = ones/np.sqrt(initial_live)
        orth[0] = ones
        num_orth = num_orth+1
    num_frozen = 0
    num_big = 0
    num_initial = 0
    while int((live*5)/4) > initial_live:
        #will have at most 1/3 * initial_live that are frozen
        #freeze up to 1/3 largest coords
        abs_y = np.absolute(y)
        #start by freezing new frozen coordinates
        is_live = (abs_y < 1.0-delta)
        diff = is_live!=was_live
        num_diff = diff.sum()
        if num_diff>0:
            for i in range(0,initial_live):
                if diff[i]:
                    e = np.zeros(initial_live)
                    e[i] = 1
                    p = make_orthogonal(orth,num_orth,e)
                    if not p is None:
                        orth[num_orth] = p
                        num_orth = num_orth + 1
                        if num_orth>=initial_live:
                            break
                    num_frozen = num_frozen + 1
            was_live = is_live
        if num_orth>=initial_live:
            break
        #may freeze same number of rows based on abs value
        num_iters = num_iters+1
        ax = a @ x
        abs_ax = np.absolute(ax)
        if num_initial < initial_live/4:
            sorted_indices = np.argsort(-abs_ax)
            for i in range(0, m):
                if was_small[sorted_indices[i]]:
                    p = make_orthogonal(orth,num_orth,b[sorted_indices[i]])
                    if not p is None:
                        orth[num_orth] = p
                        num_orth = num_orth + 1
                        if num_orth>=initial_live:
                            break
                    num_initial = num_initial + 1
                    was_small[sorted_indices[i]] = False
                    if num_initial >= initial_live/4:
                        break
        if num_orth>=initial_live:
            break
        if num_big < num_frozen:
            sorted_indices = np.argsort(-abs_ax)
            for i in range(0,m):
                if was_small[sorted_indices[i]]:
                    p = make_orthogonal(orth,num_orth,b[sorted_indices[i]])
                    if not p is None:
                        orth[num_orth] = p
                        num_orth = num_orth + 1
                        if num_orth>=initial_live:
                            break
                    num_big = num_big + 1
                    was_small[sorted_indices[i]] = False
                    if num_big >= num_frozen:
                        break
        if num_orth>=initial_live:
            break
        #sample a random vector
        g = np.random.randn(initial_live)
        #make it orthogonal to all in orth
        gamma = make_orthogonal(orth,num_orth,g)
        if gamma is None:
            break
        #set coords to zero where not live, due to numerical stability
        for i in range(0,initial_live):
            if not is_live[i]:
                gamma[i] = 0
        if np.linalg.norm(gamma,ord=np.inf)==0:
            break
        if np.inner(ax,b @ gamma) > 0:
            gamma = -gamma
        coord_mult = np.multiply(gamma,y)
        val = np.where(coord_mult < 0, (1+abs(y))/(1e-27+abs(gamma)), (1-abs(y))/(1e-27 + abs(gamma)))
        val = np.where(is_live, val, 1e27)
        z = min(val)
        y = y + z*gamma
        x[initial_is_live] = y
        new_abs_y = np.absolute(y)
        is_live = (new_abs_y < 1.0-delta)
        live = is_live.sum()
    return False


def local_improvements(a,x,time_limit,norm=np.inf, balanced=False):
    ax = a @ x
    best_norm = np.linalg.norm(ax,ord=norm)
    start_time = time.time()
    num_flip = min(7,len(x))
    if balanced and num_flip%2==1:
        num_flip = num_flip + 1
    iters = 0
    while True:
        iters = iters+1
        #Try flipping random coords
        for iteration in range(0,a.shape[1]):
            sampled_coords = np.random.randint(0,a.shape[1],num_flip)
            sampled_coords = np.sort(sampled_coords)
            allDistinct=True
            for i in range(0,num_flip-1):
                if sampled_coords[i]==sampled_coords[i+1]:
                    allDistinct=False
            if not allDistinct:
                continue
            if balanced:
                #check equal num of +1 and -1
                sum_is=0
                for i in range(0,num_flip):
                    sum_is = sum_is + x[sampled_coords[i]]
                if sum_is!=0:
                    continue
            subcols = a[:,sampled_coords]
            subx = [x[index] for index in sampled_coords]
            bx = ax - 2*(subcols @ subx)
            the_norm = np.linalg.norm(bx,ord=norm)
            if the_norm<best_norm:
                best_norm = the_norm
                for i in sampled_coords:
                    x[i] = -x[i]
                ax = bx
        if time.time()-start_time > time_limit:
            break 
    return x

def basic_local_search(a,x,norm,balanced=False):
    ax = a@x
    best_norm = np.linalg.norm(ax,ord=norm)
    improved=True
    while improved:
        improved=False
        #Try flipping single coords
        for i in range(0,a.shape[1]):
            flipped = ax - 2*x[i]*a[:,i]
            if np.linalg.norm(flipped, ord=norm) < best_norm:
                if balanced:
                    #must find one to swap with
                    for j in range(0,a.shape[1]):
                        if x[i]==x[j]:
                            continue
                        final_flipped = flipped - 2*x[j]*a[:,j]
                        if np.linalg.norm(final_flipped,ord=norm) < best_norm:
                            ax = final_flipped
                            x[i] = -x[i]
                            x[j] = -x[j]
                            best_norm = np.linalg.norm(final_flipped,ord=norm)
                            improved=True
                            break
                else:
                    ax = flipped
                    x[i] = -x[i]
                    best_norm = np.linalg.norm(flipped,ord=norm)
                    improved=True
    return x

def greedy(a,norm,balanced=False):
    x = np.zeros(a.shape[1])
    if balanced:
        so_far = np.zeros(a.shape[0])
        for i in range(0,a.shape[1]):
            if i%2==1:
                continue
            test = so_far + a[:,i]
            test_minus = so_far - a[:,i]
            if i<a.shape[1]-1:
                test = test - a[:,i+1]
                test_minus = test_minus + a[:,i+1]
            if np.linalg.norm(test,ord=norm) < np.linalg.norm(test_minus,ord=norm):
                x[i] = 1
                if i<a.shape[1]-1:
                    x[i+1]=-1
            else:
                x[i] = -1
                if i<a.shape[1]-1:
                    x[i+1] = 1
    else:
        x[0] = 1
        so_far = a[:,0]
        for i in range(1,a.shape[1]):
            test = so_far + a[:,i]
            test_minus = so_far - a[:,i]
            if np.linalg.norm(test,ord=norm) <  np.linalg.norm(test_minus,ord=norm):
                x[i] = 1
            else:
                x[i] = -1
            so_far = so_far + x[i]*a[:,i]
    return x
         
def discrepancy_minimize(a, norm=np.inf, balanced=False, local_search=0.3):
    n = a.shape[1]
    x = np.zeros(n)
    start_time = time.time()
    x = to_square(a,x,balanced)
    while not partial_color(a,x, norm, balanced):
        pass
    end_time = time.time()
    elapsed = end_time - start_time
    y = round_coloring(a,x,norm, balanced)
    y = basic_local_search(a,y,norm,balanced)
    #check if greedy approach is better
    g = greedy(a,norm,balanced)
    g = basic_local_search(a,g,norm,balanced)
    if np.linalg.norm(a@g,ord=norm) < np.linalg.norm(a@y, ord=norm):
        y = g
    y = local_improvements(a,y, elapsed*local_search, norm, balanced)
    y = basic_local_search(a,y,norm,balanced)
    total_elapsed = time.time()-start_time
    #make sure we return an int array, not floating
    z = np.zeros(n)
    for i in range(n):
        if y[i]==-1:
            z[i]=-1
        else:
            z[i]=1
    return z
