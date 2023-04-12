import numpy as np
import cv2
from PIL import Image
from math import sqrt
from matplotlib import pyplot as plt
from itertools import chain
import cv2

file = 'Bengali_Writer_Data/train_001_0001.tif'
import math
img = Image.open(file)
print(img.size)
img = np.asarray(img)
img = 255.0 - img
print(img.shape)
print(img[0])
img = img /255.0
plt.imshow(img, cmap='Greys')
plt.show()

def reductionfrom49to16(b,ii,jj,kk,count):
    nc = np.zeros((7,7,4))
    nc1 = np.zeros((9,9,4))
    app = np.zeros((64,))
    nc = b
    h=0
    for i in range(0,7,2):
        for j in range(0,7,2):
            for k in range(4):
                nc[i][j][k] = nc[i][j][k]*2
    
    for i in range(1,8):
        for j in range(1,8):
            for k in range(4):
                nc1[i][j][k] = nc[i-1][j-1][k]
    
    
    for i in range(1,8,2):
        for j in range(1,8,2):
            for k in range(4):
                nc2[int(i/2)][int(j/2)][k]=nc1[i][j][k]+ nc1[i-1][j-1][k]+nc1[i-1][j][k]+nc1[i-1][j+1][k]+nc1[i][j-1][k]+nc1[i][j+1][k]+nc1[i+1][j-1][k]+nc1[i+1][j][k]+nc1[i+1][j+1][k]
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                nc2[i][j][k]=nc2[i][j][k]/count
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                app[h]=nc2[i][j][k]
                h = h+1
    return app


def least(arr):
    m = arr[0]
    for i in range(300):
        if arr[i]<m:
            m=arr[i]
    return m

def greatest(arr):
    m = 0
    for i in range(300):
        if arr[i]>m:
            m=arr[i]
    return m

def division64(k,a,dim):
    j=0
    prev=0
    p = k/dim
    for i in range(0,dim):
        j = j+p
        q=j
        t = q+0.5
        if j>t:
            a[i] = q+1-prev
        else:
            a[i] = q-prev
        prev = prev + a[i]
    return a
    
    
def division(k,a,dim):
    j,prev = 0,0
    p = k/dim
    print('IN division function',p)
    for i in range(dim):
        j = j+p
        q = j
        t = q+0.5
        if j>t:
            a[i] = q+1-prev
        else:
            a[i] = q-prev
        prev = prev + a[i]
    return a
a = img
print(a.shape,np.max(a))
b1 = a
c1,c2,c3,c4 = np.zeros(2300,),np.zeros(2300,),np.zeros(2300,),np.zeros(2300,)
l1,l2,l3,l4 = 0,0,0,0
for i in range(2300):
    c1[i] = 999
    c2[i] = 0
    c3[i] = 999
    c4[i] = 0
u = col-1
y1 = row-1
b = np.zeros((7,7,4))
for i in range(row):
    for j in range(col):
        if a[i][j]==1:
            c1[l1] = j
            l1 = l1+1
            break
h1 = least(c1)
print('h1 is:',h1)
for i in range(row):
    for j in range(u,-1,-1):
        if a[i][j]==1:
            c2[l2] = j
            l2 = l2+1
            break
h2 = greatest(c2)
print('h2 is:',h2)
for i in range(col):
    for j in range(row):
        if a[j][i]==1:
            c3[l3]=j
            l3 = l3+1
            break

h3 = least(c3)
print('h3 is:',h3)
for i in range(col):
    for j in range(y1,-1,-1):
        if a[j][i]==1:
            c4[l4] = j
            l4 = l4+1
            break

h4 = greatest(c4)
print('h4 is:',h4)
h1,h2,h3,h4 = int(h1),int(h2),int(h3),int(h4)

ro = h4-h3+1
co = h2-h1+1
print(h1,h2,h3,h4)
r = np.zeros((7,))
c = np.zeros((7,))
r = division64(ro,r,7)
c = division64(co,c,7)
#print('r is:',r)
#print('c is:',c)
h8 = h3+1
h9 = h1 +1
for i in range(h8,h4,1):
    for j in range(h9,h2,1):
        if a[i][j]==0:
            b1[i][j] = 0
        else:
            if a[i][j]==1:
                if a[i][j+1]==1 and a[i-1][j+1]==1 and a[i-1][j]==1 and a[i-1][j-1]==1 and a[i][j-1]==1 and a[i+1][j-1]==1 and a[i+1][j]==1 and a[i+1][j+1]==1:
                    b1[i][j] = 1
                else:
                    b1[i][j] = 3

for i in range(h3,h4+1):
    if a[i][h1]==0:
        b1[i][h1] = 0
    else:
        b1[i][h1] = 3
    if a[i][h2]==0:
        b1[i][h2]=0
    else:
        b1[i][h2] = 3
    
for j in range(h1,h2+1):
    if a[h3][j]==0:
        b1[h3][j] = 0
    else:
        b1[h3][j] = 3
    if a[h4][j] ==0:
        b1[h4][j] = 0
    else:
        b1[h4][j] = 3

for m in range(0,7):
    for n in range(0,7):
        s = h1
        t=0
        y=h3
        z=0
        for i in range(m):
            y=y+r[i]
        z = y+r[m]
        for j in range(0,n):
            s = s+c[j]
        t = s+c[n]
        #print(z,y,t,s)
        z,y,t,s = int(z),int(y),int(t),int(s)
        for i in range(y,z):
            for j in range(s,t):
                if b1[i][j]==0:
                    continue
                elif b1[i][j]==1:
                    continue
                else:
                    if (j+1)<t:
                        if b1[i][j+1]==3:
                            b[m][n][0] = b[m][n][0]+1
                    if (i-1)>=y and (j+1)<t:
                        if b1[i-1][j+1]==3:
                            b[m][n][1] = b[m][n][1]+1
                    if (i-1)>=y:
                        if b1[i-1][j]==3:
                            b[m][n][2] = b[m][n][2]+1
                    if (i-1)>=y and (j-1)>=s:
                        if b1[i-1][j-1]==3:
                            b[m][n][3] = b[m][n][3]+1
 







count = 0
for i in range(h3,h4+1):
    for j in range(h1,h2+1):
        if b1[i][j]==3:
            count = count+1

app = reductionfrom49to16(b,7,7,4,count)
b = np.asarray(b)
print(b.shape)
for i in range(64):
    if app[i]>0:
        flag = 1
app = np.asarray(app)
print(app.shape)
print(app)


