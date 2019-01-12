import numpy as np

def hilbertCurve(order):
    for i in range(0,order):
        if i == 0:
            curve = np.array([[0,1],[3,2]])
        else:
            offset = 4**(i)
            q1 = np.rot90(np.flip(curve,0),-1)
            q2 = curve + offset
            q3 = curve + offset*2
            q4 = np.rot90(np.flip(curve,1),-1) + offset*3
            top = np.concatenate((q1,q2),axis=1)
            bot = np.concatenate((q4,q3),axis=1)
            curve = np.concatenate((top,bot))
    return(curve)

#def hilbertCurve(x, y):
 #   return

if __name__ == '__main__':
    import sys
    try:
        order = int(sys.argv[1])
    except:
        print 'usage: {order}'
        exit()
    if order < 1:
        print 'Error: error must be greater than 0'
        exit()
    else:
        curve = hilbertCurve(order)
        print curve

