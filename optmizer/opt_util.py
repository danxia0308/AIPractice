import numpy as np

def cal_adam_paper(m,v,g,var,t=0,beta1=0.9,beta2=0.99,epsilon=1e-8,lr=0.8):
    t=t+1
    alpha=lr*np.sqrt(1-np.power(beta2,t))/(1-np.power(beta1,t))
#     alpha=0.001
    print("np.sqrt(1-beta1)/(1-beta1)=",np.sqrt(1-beta1)/(1-beta1))
#     alpha=0.01
    m = m +(g-m)*(1-beta1)
    v = v +(np.square(g)-v)*(1-beta2)
    m1 = m/(1-np.power(beta1,t))
    v1 = v/(1-np.power(beta2,t))
    var = var - m1*alpha/(np.sqrt(v1)+epsilon)
    print("alpha={}, m={},v={},var={}".format(alpha, m,v,var))
#     print("m1={},v1={},np.sqrt(v1)={},m1/(np.sqrt(v1)+epsilon={}".format(m1,v1,np.sqrt(v1),m1/(np.sqrt(v1)+epsilon)))

def cal_adam(m,v,g,var,t=0,beta1=0.9,beta2=0.99,epsilon=1e-8,lr=0.8):
    t=t+1
    alpha=lr*np.sqrt(1-np.power(beta2,t))/(1-np.power(beta1,t))
#     alpha=0.001
    print("np.sqrt(1-beta1)/(1-beta1)=",np.sqrt(1-beta1)/(1-beta1))
#     alpha=0.01
    m = m +(g-m)*(1-beta1)
    v = v +(np.square(g)-v)*(1-beta2)
    m1 = m/(1-np.power(beta1,t))
    v1 = v/(1-np.power(beta2,t))
    var = var - m1*alpha/(np.sqrt(v1)+epsilon)
#     var = var - ((g * (1 - beta1) + beta1 * m) * alpha) /(np.sqrt(v) + epsilon);
    print("alpha={}, m={},v={},var={}".format(alpha, m,v,var))
#     print("m1={},v1={},np.sqrt(v1)={},m1/(np.sqrt(v1)+epsilon={}".format(m1,v1,np.sqrt(v1),m1/(np.sqrt(v1)+epsilon)))

# cal_adam(-22.567665, 50.929268, -225.67659, -0.3347394)
# cal_adam(0, 0, -225.67659, -1.1347394,0)
# cal_adam(-22.567658999999995, 50.92992327402815, -110.57677, -0.3347394,1)


def func(t, beta1=0.9,beta2=0.999):
    return np.sqrt(1-np.power(beta2,t))/(1-np.power(beta1,t))

for i in range(9000,20000):
    print('i={},{}'.format(i,func(i)))
