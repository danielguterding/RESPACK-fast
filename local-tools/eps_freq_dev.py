import sys, numpy as np
def rd(p):
    with open(p,'rb') as f:
        n=np.fromfile(f,np.int32,1)[0]; d=np.fromfile(f,np.complex128,n//16)
    return d
T=sys.argv[1]; q=sys.argv[2]; fk=sys.argv[3]; nen=70
a=rd(f'{T}/run-ref/dir-eps/{q}/dat.epsqw.600'); b=rd(f'{T}/run-opt/dir-eps/{q}/dat.epsqw.600'); c=rd(f'{fk}/dir-eps/{q}/dat.epsqw.600')
ng=int(round(np.sqrt(a.size/nen)))
A=a.reshape(nen,ng,ng); B=b.reshape(nen,ng,ng); C=c.reshape(nen,ng,ng)
em=np.loadtxt(f'{T}/run-opt/dir-eps/dat.wgrid',skiprows=1)[:,0]*27.21151
print(f'{q}: NG={ng}')
print(' ie   w(eV)   max|eps-1|   rel dev (O2 vs O3)   rel dev (O3 vs fork)')
for ie in range(nen):
    s=np.abs(A[ie]).max()
    print(f'{ie+1:3d} {em[ie]:8.3f} {s:10.3e}   {np.abs(A[ie]-B[ie]).max()/s:10.2e}        {np.abs(B[ie]-C[ie]).max()/s:10.2e}')
