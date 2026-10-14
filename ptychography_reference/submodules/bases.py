import numpy as np

def is_prime(n):
  for i in range(2,int(np.sqrt(n))+1):
    if (n%i) == 0:
      return False
  return True

def row_constructer(n):
    x = (np.arange(n)**2) % n
    chi = np.where(np.isin(np.arange(n), x), 1, -1)
    chi[0] = 0
    return chi

def jacobsthal(n):
    q = np.zeros((n,n))
    row = row_constructer(n)
    q[0] = row
    for i in range(1,n):
        q[i,i:] = row[:-1*i]
        q[i,:i] = row[-1*i:] 
    return q.T

def paley_1(n):
    q = jacobsthal(n-1)
    h = np.zeros((n,n))
    h[1:,0] = -1
    h[0,1:] = 1
    h[1:,1:] = q
    return h + np.eye(n)

def power2hadamard(n):
    logn = np.log2(n).astype('int')
    element = 1
    h = 1
    for i in range(1,logn+1):
        l2 = 2**i // 2
        h = np.zeros((2*l2, 2*l2))
        h[:l2,:l2] = element
        h[:l2,l2:] = element
        h[l2:,:l2] = element
        h[l2:,l2:] = -1*element
        element = h
    return h

def hadamard36():
    H1 = "++++++++++++++++++-+++++++++++++++++ ++++-+---++---+-+++-++-+---++---+-++ +++++-+---++---+-+++-++-+---++---+-+ ++++++-+---++---+-+++-++-+---++---+- +-+++++-+---++---++-++-++-+---++---+ ++-+++++-+---++---++-++-++-+---++--- +-+-+++++-+---++--+-+-++-++-+---++-- +--+-+++++-+---++-+--+-++-++-+---++- +---+-+++++-+---+++---+-++-++-+---++ ++---+-+++++-+---+++---+-++-++-+---+ +++---+-+++++-+---+++---+-++-++-+--- +-++---+-+++++-+--+-++---+-++-++-+-- +--++---+-+++++-+-+--++---+-++-++-+- +---++---+-+++++-++---++---+-++-++-+ ++---++---+-+++++-++---++---+-++-++- +-+---++---+-++++++-+---++---+-++-++ ++-+---++---+-++++++-+---++---+-++-+ +++-+---++---+-++++++-+---++---+-++- -+++++++++++++++++------------------ +-++-+---++---+-++----+-+++--+++-+-- ++-++-+---++---+-+-----+-+++--+++-+- +++-++-+---++---+-------+-+++--+++-+ +-++-++-+---++---+-+-----+-+++--+++- ++-++-++-+---++-----+-----+-+++--+++ +-+-++-++-+---++---+-+-----+-+++--++ +--+-++-++-+---++--++-+-----+-+++--+ +---+-++-++-+---++-+++-+-----+-+++-- ++---+-++-++-+---+--+++-+-----+-+++- +++---+-++-++-+------+++-+-----+-+++ +-++---+-++-++-+---+--+++-+-----+-++ +--++---+-++-++-+--++--+++-+-----+-+ +---++---+-++-++-+-+++--+++-+-----+- ++---++---+-++-++---+++--+++-+-----+ +-+---++---+-++-++-+-+++--+++-+----- ++-+---++---+-++-+--+-+++--+++-+---- +++-+---++---+-++----+-+++--+++-+---"
    H1 = list(H1)
    H_arr = np.zeros((36,36))
    i=0
    j=0
    for ele in H1:
        if ele=='+':
            H_arr[i][j] = 1
            j+=1
        elif ele=='-':
            H_arr[i][j] =-1
            j+=1
        else:
            i+= 1
            j=0
    return H_arr

def hadamard_null():
    return np.zeros_like(hadamard36()) 

def hadamard(n):
    if n==0:
        return hadamard_null()
    elif np.log2(n)==np.log2(n).astype('int'):
        return power2hadamard(n)
    elif ((n-1) % 4 == 3) and is_prime(n-1):
        return paley_1(n)
    elif ((n-1) % 4 == 1) and is_prime(n-1):
        print('paley_2 not yet implemented')
        return -1
    elif n==36:
        return hadamard36()
    else:
        print('not a possible rank for hadamard matrix')
        return -1

def triangular(n):
    p = np.zeros((3,n),'complex')
    for i in range(n):
        p[1,i] = np.pi*2/3 * (-1)**i
        p[2,i] = np.pi*1/3 * (-1)**i
    return np.exp(1j * p)
