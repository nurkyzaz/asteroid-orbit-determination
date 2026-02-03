
import numpy as np
from math import *
from odlib_nurkyz import *

file = open('2021MoGtestinput.txt')
lines = file.readlines()
splitList = []
wantedJulian = JD_time(2021, 7, 24, 7, 0, 0)

print('Possible Observations:')

#Taking in user input
for i in range(len(lines)):
    placeHolder = lines[i][:-1]
    splitList.append(placeHolder)
    print(f'({i+1}) {placeHolder}')

if(len(lines) > 3):
    print('Please choose 3 of the previous options')
    lineOpt1, lineOpt2, lineOpt3 = [int(input(f'{label} Observation: ')) for label in ['First', 'Middle', 'Final']]
    split1, split2, split3 = [splitList[lineOpt-1] for lineOpt in [lineOpt1, lineOpt2, lineOpt3]]

else:
    split1, split2, split3 = splitList[:3]

def str_to_int(line):
    split = line.split(' ')
    hhmmss = split[3].split(':')
    hh = int(hhmmss[0])
    mm = int(hhmmss[1])
    ss = float(hhmmss[2])
    JD = JD_time(int(split[0]), int(split[1]), int(split[2]), hh, mm, ss)

    RA1 = split[4].split(':')
    ra = (int(RA1[0]) + int(RA1[1])/60 + float(RA1[2])/3600)*360/24
    
    Dec1 = split[5].split(':')
    dec = angleConverter(int(Dec1[0]),int(Dec1[1]),float(Dec1[2]))
    
    sun= np.array([float(split[6]), float(split[7]), float(split[8])])

    return JD, radians(ra), radians(dec), sun

JD1, RA1, dec1, sun1 = str_to_int(lines[0][:-1])
JD2, RA2, dec2, sun2 = str_to_int(lines[1][:-1])
JD3, RA3, dec3, sun3 = str_to_int(lines[2][:-1])

rhohat1 = get_rhohat(RA1, dec1)
rhoHat2 = get_rhohat(RA2, dec2)
rhoHat3 = get_rhohat(RA3, dec3)

def getD(rho1, rho2, rho3, sun1, sun2, sun3):
    d0 = np.dot(rho1, np.cross(rho2, rho3))
    d21 = np.dot(np.cross(rho1, sun1), rho3)
    d22 = np.dot(np.cross(rho1, sun2), rho3)
    d23 = np.dot(np.cross(rho1, sun3), rho3)
    return np.array([d0, d21, d22, d23])

tau1 = (JD1-JD2)/(365.25/(2*pi))
tau3 = (JD3-JD2)/(365.25/(2*pi))
mainTau = tau3-tau1
tauList = [tau1, tau3, mainTau]

dlist = getD(rhohat1, rhoHat2, rhoHat3, sun1, sun2, sun3)
d0 = dlist[0]

d11 = np.dot(np.cross(sun1, rhoHat2), rhoHat3)
d12 = np.dot(np.cross(sun2, rhoHat2), rhoHat3)
d13 = np.dot(np.cross(sun3, rhoHat2), rhoHat3)
d21 = dlist[1]
d22 = dlist[2]
d23 = dlist[3]
d31 = np.dot(rhohat1,np.cross(rhoHat2, sun1))
d32 = np.dot(rhohat1,np.cross(rhoHat2, sun2))
d33 = np.dot(rhohat1,np.cross(rhoHat2, sun3))

#rotation matrix
rotAngle = 0.409022397115
rotationMatrix = [[1,0,0],[0,cos(rotAngle), sin(rotAngle)], [0,-sin(rotAngle), cos(rotAngle)]]
ar_rotation = np.array(rotationMatrix)

#Finding the positive roots and the rho values
realPosRoots, rhoVals = lagrange(tauList, sun2, rhoHat2, dlist)
true_roots = []
true_rho_vs = []

#Finding true rho values
for i in range(len(rhoVals)):
    if rhoVals[i] >= 0:
        true_rho_vs.append(rhoVals[i])
        true_roots.append(realPosRoots[i])

print('Possible Roots')
for i in range(len(true_roots)):
    print(f'({i+1}) r2 = {true_roots[i]} AU (rho2 = {true_rho_vs[i]} AU)')

#Taking input for rhos
if(len(true_rho_vs) > 1):
    print('Please choose one of the previous roots: ')
    optionRho = int(input())
    trueRho = true_rho_vs[optionRho + 1]
    trueRoots = true_roots[optionRho + 1]
else:
    trueRho = true_rho_vs[0]
    trueRoots = true_roots[0]
    print(f'Only 1 reasonable root calculated. Using r2 = {trueRoots} AU')

#Find initial value for the f ang g vectors
mu = 1
f1 = 1 - (mu/(2*trueRoots**3))*tau1**2
f3 = 1 - (mu/(2*trueRoots**3))*tau3**2
g1 = tau1 - (mu/(6*trueRoots**3))*tau1**3
g3 = tau3 - (mu/(6*trueRoots**3))*tau3**3

c1 = g3/(f1*g3-g1*f3)
c2 = -1
c3 = -g1/(f1*g3-g1*f3)

rho1 = (c1*d11 + c2*d12 + c3*d13)/(c1*d0)
rho2 = (c1*d21 + c2*d22 + c3*d23)/(c2*d0)
rho3 = (c1*d31 + c2*d32 + c3*d33)/(c3*d0)

r1 = rho1*rhohat1 - sun1
r2 = rho2*rhoHat2 - sun2
r3 = rho3*rhoHat3 - sun3

d1 = -f3/(f1*g3-f3*g1)
d3 = f1/(f1*g3-f3*g1)

print('')
print('Main Iteration:')
new_rho = abs(trueRho - rho2)
r2dot_v = d1*r1 + d3*r3

count = 0
tolerance = 1E-12
cpeed = 173.144643267
while(new_rho > tolerance):
    lightJD1 = JD1 - rho1/cpeed
    lightJD2 = JD2 - rho2/cpeed
    lightJD3 = JD3 - rho3/cpeed

    tau1 = (lightJD1-lightJD2)/(365.25/(2*pi))
    tau3 = (lightJD3-lightJD2)/(365.25/(2*pi))
    mainTau = tau3-tau1
    tauList = [tau1, tau3, mainTau]

    trueRho = rho2
    f1, f3, g1, g3 = fg(tau1, tau3, r2, r2dot_v, flag = 0)

    c1 = g3/(f1*g3-g1*f3)
    c2 = -1
    c3 = -g1/(f1*g3-g1*f3)

    rho1 = (c1*d11 + c2*d12 + c3*d13)/(c1*d0)
    rho2 = (c1*d21 + c2*d22 + c3*d23)/(c2*d0)
    rho3 = (c1*d31 + c2*d32 + c3*d33)/(c3*d0)

    r1 = rho1*rhohat1 - sun1
    r2 = rho2*rhoHat2 - sun2
    r3 = rho3*rhoHat3 - sun3

    d1 = -f3/(f1*g3-f3*g1)
    d3 = f1/(f1*g3-f3*g1)

    new_rho = abs(trueRho - rho2)
    r2dot_v = d1*r1 + d3*r3

    lightChange = rho2/cpeed

    count += 1

    print(f'({count}) Change in rho2 = {new_rho} AU; light travel time = {lightChange*3600*24} sec')


rotatedR2 = np.matmul(ar_rotation, r2)
rotatedR2Dot = np.matmul(ar_rotation, r2dot_v)

print(f'In {count} iterations, r2 and r2dot converged to:')
print(f'r2 = {r2} = {mag(r2)} AU')
print(f'r2dot = {r2dot_v} = {mag(r2dot_v)*(2*pi/365)} AU/day')
print(f'r2 = {rotatedR2} = {mag(rotatedR2)} AU')
print(f'r2dot = {rotatedR2Dot} = {mag(rotatedR2Dot)*(2*pi/365)} AU/day')
print(f'with rho2 = {rho2} AU')
print('*'*20)

final_elements(rotatedR2, rotatedR2Dot, JD2, wantedJulian)