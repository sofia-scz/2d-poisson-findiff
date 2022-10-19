import numpy as np

### set up del sistema de ecuaciones

# esta función setea la matriz de coeficientes y el vector independiente
# main setup
def setup_main(dx,dy,n,m,xi,F):
    
    cx,cy = -dx**2/2/(dx**2+dy**2),-dy**2/2/(dx**2+dy**2) # constantes
    Mdim = (n+1)*(m+1)  # dimension de la matriz y vec indep
    M_coef,V_indep = np.zeros((Mdim,Mdim)),np.zeros(Mdim) # crear mat y v indep

    for i in range(1,n):      # barrer sobre los puntos internos de la grilla
        for j in range(1,m):
            k = aux_ij2k(i,j,m)  # pasar i,j a k
            
            V_indep[k] = dx**2*cy*xi[i,j]  # el k-esimo elem del v indep
                                           # es xi(i,j) por estas constantes
            M_coef[k,k] = 1         # coef central, c(i,j) = 1
            M_coef[k,k+(m+1)],M_coef[k,k-(m+1)] = cx,cx  # coefs celdas ady hori
            M_coef[k,k+1],M_coef[k,k-1] = cy,cy # coefs celdas ady vert

    # barrer sobre la frontera
    for j in range(m+1):  # barrer sobre x con y=y0,y=yf
        
        k = aux_ij2k(0,j,m)
        V_indep[k] = F[0,j]
        M_coef[k,k] = 1         # coef 
        
        k = aux_ij2k(n,j,m)
        V_indep[k] = F[n,j]
        M_coef[k,k] = 1         # coef central
        
    for i in range(n+1):  # barrer sobre y con x=x0,x=xf
        
        k = aux_ij2k(i,0,m)
        V_indep[k] = F[i,0]
        M_coef[k,k] = 1        # coef 
        
        k = aux_ij2k(i,m,m)
        V_indep[k] = F[i,m]
        M_coef[k,k] = 1        # coef central
        
        
    return M_coef,V_indep

### funciones auxiliares

# pasar i,j -> k
def aux_ij2k(i,j,m):
    return i*(m+1)+j

# pasar k -> i,j
def aux_k2ij(k,m):
    i = k//(m+1)
    return i,k-i*(m+1)

# pasar phi i,j -> k   (obsoleta?)
def aux_phi_ij2k(phi,n,m):
    phik = np.zeros((n+1)*(m+1))
    for i in range(n):
        for j in range(m):
            k = aux_ij2k(i,j,m)
            phik[k] = phi[i,j]
    return phik

# pasar phi k -> i,j  
def aux_phi_k2ij(phik,n,m):
    phi = np.zeros((n+1,m+1))
    for i in range(n+1):
        for j in range(m+1):
            k = aux_ij2k(i,j,m)
            phi[i,j] = phik[k]
    return phi

### main

# esta funcion toma los parametros de la región, grilla, condiciones de borde
# y fuente, y llama la función que crea el sistema de ecuaciones, lo resuelve y
# llama a la función que transforma el resultado a la representación de matriz

def main(x0,xf,y0,yf,n,m,xi,F):
    # definir constantes
    dx,dy = (xf-x0)/m,(yf-y0)/n
    
    # armar sistema de ecuaciones
    M_coef,V_indep = setup_main(dx,dy,n,m,xi,F)
    
    # invertir la matriz y aplicarla al vector indep
    Minv = np.linalg.inv(M_coef)
    sol = np.matmul(Minv,V_indep) # solucion en formato k (cambiar esto por linalg.solve?)

    return aux_phi_k2ij(sol,n,m) # devolver solucion en formato i,j
