import numpy as np
import numpy.polynomial.polynomial as polyroots
from astropy.io import fits
import matplotlib.pyplot as plt
from math import *
import scipy.optimize as op

def JD_time(year, month, day, hours, minutes, seconds):
    #from current date in UT to Julian Date
    UT = hours + minutes/60 + seconds/3600
    J = 367*year - int((7*(year+int((month+9)/12)))/(4)) + int((275*month)/9)+ day + 1721013.5
    jD = J + UT/24
    return jD

def readfile(fileName):
    file = open(fileName)
    list = file.readlines()
    for i in list:
        if(i[0] == 'X'):
            X = float(i[4:-1])
        elif(i[0] == 'Y'):
            Y = float(i[4:-1])
        elif(i[0] == 'Z'):
            Z = float(i[4:-1])
        elif('VX' in i):
            VX = float(i[4:-1]) *365.2563835/(2*pi)
        elif('VY' in i):
            VY = float(i[4:-1]) *365.2563835/(2*pi)
        elif('VZ' in i):
            VZ = float(i[4:-1]) *365.2563835/(2*pi)
    r_V = [X,Y,Z]
    r_dot_V = [VX,VY,VZ]
    return r_V, r_dot_V

def orbitalElements(fileName, jDate, true_a=1, true_e=1, true_i=1, true_long_a=1, true_per=1, true_ma=1, true_time_from_per=1):
    rV, r_dot_V = readfile(fileName)
    h, a, e, i, longAsc, peri = get_orbital_e(rV, r_dot_V)
    r = mag(rV)
    ea = acos((1/e)*(1-(r/a)))
    ma = ea - e*sin(ea)
    k = .01720209895
    perihelion_t = -((ma/sqrt(k**2/a**3))-jDate)

    print('Orbital Elements of Asteroid')
    print('--'*20)
    print(f"Angular Momentum (h): ({h[0]}, {h[1]}, {h[2]})")
    print(f"Semimajor axis (a): {a}")
    print(f'Expected (a): {true_a}\t\tError: {find_error(a,true_a)}%')
    print(f'Eccentricity (e): {e}')
    print(f'Expected (e): {true_e}\t\tError: {find_error(e,true_e)}%')
    print(f"Inclination (i): {i*180/pi}")
    print(f'Expected (i): {true_i*180/pi}\t\tError: {find_error(i,true_i)}%')
    print(f'Longitudinal Ascension (capital Omega): {longAsc}')
    print(f'Expected (capital Omega): {true_long_a}\tError: {find_error(longAsc,true_long_a)}%')
    print(f"Argument of periapsis (lowercase Omega): {peri}")
    print(f'Expected (lowercase Omega): {true_per}\tError: {find_error(peri,true_per)}%')
    print(f'Mean Anomaly (M): {ma*180/pi}')
    print(f'Expected (M): {true_ma*180/pi}\tError: {find_error(ma, true_ma)}%')
    print(f'Julian Date: {jDate}')
    print(f'Time since perihelion: {jDate-perihelion_t}')
    print(f'Expected time: {jDate-true_time_from_per}\tError: {find_error(jDate-perihelion_t, jDate-true_time_from_per)}%')
    print('--'*20)

    return rV, r_dot_V, h, a, e, i, longAsc, peri, ma, perihelion_t


def find_error(real, expected):
    return abs(real-expected) / real * 100

def newton_method(f, fPrime, tolerance = .00000001, scipy = False, startingVal = 1):
    if scipy:
        Enull = op.newton(f, startingVal)
        return Enull
    count = 0
    Enull = 1
    trial = f(startingVal)
    while(abs(trial)>= tolerance):
        count += 1
        Enull = Enull - (f(Enull)/fPrime(Enull))
        trial = f(Enull)
    return Enull

def celestCoord(fileName, julian_date, true_a, true_e, true_i, true_long_a, true_per, true_MA, true_time_from_per, true_ra, true_dec):
    #Getting the orbital elements
    rVec, rDotVec, h, a, e, i, longAsc, peri, meanAn, perihelionTime = orbitalElements(fileName, julian_date, true_a, true_e, true_i, true_long_a, true_per, true_MA, true_time_from_per)

    #Getting E
    new_julian = JD_time(2018, 8, 3, 0, 0, 0) #Time of the date where we want to end up
    k = .01720209894
    mu = 1
    n  = k*sqrt(mu/a**3)
    newMeanAn = (meanAn - n*(julian_date-new_julian))
    E = newton_method(1, newMeanAn, e, tolerance=.00001)

    incartesian = np.array([a*cos(E)-a*e, a*sqrt(1-e**2)*sin(E), 0])
    #Rotations
    small_omega_v = np.array([[cos(peri), -sin(peri), 0], [sin(peri), cos(peri), 0],[0,0,1]])  #by periapsis
    #by inclination
    inc_v = np.array([[1,0,0],[0,cos(i), -sin(i)], [0, sin(i), cos(i)]])
    #by longitudinal ascention
    big_omega_v = np.array([[cos(longAsc), -sin(longAsc), 0],[sin(longAsc), cos(longAsc), 0],[0,0,1]])

    rot1 = np.matmul(small_omega_v, incartesian)
    rot2 = np.matmul(inc_v, rot1)
    rot3 = np.matmul(big_omega_v, rot2)

    #by the ecliptic
    ecliptic = radians(23.44)
    ecliptic_v = np.array([[1,0,0],[0,cos(ecliptic), -sin(ecliptic)],[0,sin(ecliptic), cos(ecliptic)]])
    rotation_v = np.matmul(ecliptic_v, rot3)

    #Earth-sun vector 
    earth_sun_v = np.array([-6.574011189521245E-01, 7.092445973782825E-01, 3.074588267894852E-01])

    #Getting the range vector
    range_v = rotation_v + earth_sun_v
    mag_rho = range_v/mag(range_v)

    #Find RA and Dec
    dec = asin(mag_rho[2])
    cos_ra = mag_rho[0]/cos(dec)
    sin_ra = mag_rho[1]/cos(dec)
    RA = radians(quadFinding(cos_ra, sin_ra))

    print(f'RA: {RA*180/pi} degress')
    print(f'Expected RA: {true_ra} degrees')
    print(f'RA Error: {find_error(RA, radians(true_ra))}%')
    print('--'*20)
    print(f'Declination: {dec*180/pi} degrees')
    print(f'Expected Declination: {true_dec} degrees')
    print(f'Declination Error: {find_error(dec, radians(true_dec))}%')
    print('--'*20)
    return RA, dec

def lagrange(tau_list, sun_v, rho_hat, D): #Gets roots using the scalar equation of Lagrange
    #Getting Ds
    d0 = D[0]
    d21 = D[1]
    d22 = D[2]
    d23 = D[3]
    #Getting taus
    big_tau = tau_list[2]
    tau1 = tau_list[0]
    tau3 = tau_list[1]
    aOne = tau3/big_tau
    bOne = (aOne/6)*(big_tau**2-tau3**2)
    aThree = -tau1/big_tau
    bThree = (aThree/6)*(big_tau**2-tau1**2)

    A = (aOne*d21 - d22 + aThree*d23)/(-d0)
    B = (bOne*d21 + bThree*d23)/(-d0)
    F = sun_v[0]**2 + sun_v[1]**2 + sun_v[2]**2
    E = -2*(np.dot(rho_hat, sun_v))

    a = -(A**2 + E*A + F)
    b = -(2*A*B + B*E)
    c = -B**2

    roots = polyroots.polyroots([c,0, 0, b, 0, 0, a, 0, 1])

    #Finds the real roots of the function
    true_roots = np.real(roots)
    true_rho_roots = []
    rho_values = []
    for i in true_roots:
        if i > 0:
            true_rho_roots.append(i)
            rho_values.append(A + B/i**3)

    return true_rho_roots, rho_values

def fg(tau1, tau3, r2, r2Dot, flag = 0, tolerance = 1E-12):
    rMag = mag(r2)
    u = 1/rMag**3
    z = (np.dot(r2, r2Dot))/rMag**2
    q = (np.dot(r2Dot, r2Dot))/rMag**2 - u
    a = (2/mag(r2) - mag(r2Dot)**2)**(-1)
    n = np.sqrt(1/a**3)
    h = momentum(r2, r2Dot)
    magH = mag(h)
    e = sqrt(1-((magH**2)/a))

    plist = np.dot(r2, r2Dot)/(n*a**2)
    get_x1 = plist*cos(n*tau1 - plist) + (1-rMag/a)*sin(n*tau1 - plist)
    get_xp1 = get_x1>= 0
    get_x3 = plist*cos(n*tau3 - plist) + (1-rMag/a)*sin(n*tau3 - plist)
    get_xp3 = get_x3>= 0

    if flag == 3:
        f1 = 1 - .5*u*tau1**2 + .5*u*z*tau1**3
        g1 = tau1 - (u*tau1**3)/6
        f3 = 1 - .5*u*tau3**2 + .5*u*z*tau3**3
        g3 = tau3 - (u*tau3**3)/6
    elif flag == 4:
        f1 = 1 - .5*u*tau1**2 + .5*u*z*tau1**3 + ((3*u*q-15*u*z**2+u**2)*tau1**4)/24
        g1 = tau1 - (u*tau1**3)/6 + (u*z*tau1**4)/4
        f3 = 1 - .5*u*tau3**2 + .5*u*z*tau3**3 + ((3*u*q-15*u*z**2+u**2)*tau3**4)/24
        g3 = tau3 - (u*tau3**3)/6 + (u*z*tau3**4)/4
    elif flag == 0:
        #fnewton-raphson
        def f1Func(x):
            return x - (1-mag(r2)/a)*sin(x) + (plist)*(1-cos(x)) - n*tau1
        def f3Func(x):
            return x - (1-mag(r2)/a)*sin(x) + (plist)*(1-cos(x)) - n*tau3
        def fPrime(x):
            return 1 - (1-rMag/a)*cos(x) + plist*sin(x)
        
        #first value for newton-raphson
        if(e <= .1):
            xNull1 = n*tau1
            xNull3 = n*tau3
        else:
            if(get_xp1):
                xNull1 = n*tau1 + .85*e - plist
            else:
                xNull1 = n*tau1 - .85*e - plist
            if(get_xp3):
                xNull3 = n*tau3 + .85*e - plist
            else:
                xNull3 = n*tau3 - .85*e - plist

        deltaE1 = newton_method(f1Func, fPrime, scipy = True, startingVal = xNull1, tolerance=tolerance)
        deltaE3 = newton_method(f3Func, fPrime, scipy = True, startingVal = xNull3, tolerance=tolerance)

        f1 = 1 - (a/rMag)*(1-cos(deltaE1))
        f3 = 1 - (a/rMag)*(1-cos(deltaE3))
        g1 = tau1 + (1/n)*(sin(deltaE1) - deltaE1)
        g3 = tau3 + (1/n)*(sin(deltaE3) - deltaE3)
    return f1, f3, g1, g3

def photometry(fileName, pixelVal, radius, innerAn, outerAnn, readNoise = 15, darkCurrent = .02):
    data = fits.getdata(fileName)
    centerX = pixelVal[0]
    centerY = pixelVal[1]
    center = centroid(data, pixelVal[0], pixelVal[1], radius)
    aperture = data[centerY-radius:centerY + radius +1,centerX-radius:centerX+radius+1]
    apertureCircle = []

    #Finding the ADU_ap
    tempCenter = radius + 1

    for i in range(aperture.shape[0]):
        for j in range(aperture.shape[1]):
            yVal = i
            xVal = j
            if(((xVal-tempCenter)**2 + (yVal-tempCenter)**2) <= radius**2):
                apertureCircle.append(aperture[j][i])
    
    apCircleArr = np.array(apertureCircle)
    sumCircle = np.sum(apCircleArr)

    #Finding the ADU_an
    tempCenter = outerAnn + 1
    annulus = data[centerY-outerAnn:centerY + outerAnn +1,centerX-outerAnn:centerX+outerAnn+1]
    annulusCircle = []
    for i in range(annulus.shape[0]):
        for j in range(annulus.shape[1]):
            yVal = i
            xVal = j
            if((((xVal-tempCenter)**2 + (yVal-tempCenter)**2) <= outerAnn**2) and (((xVal - tempCenter)**2 + (yVal - tempCenter)**2) >= innerAn**2)):
                annulusCircle.append(annulus[j][i])

    annArr = np.array(annulusCircle)
    medianAn = np.median(annArr)
    nAP = len(apertureCircle)
    nAN = len(annulusCircle)

    signal = sumCircle - medianAn*nAP

    mInst = -2.5*log10(signal)
    SNR = signal/(sqrt(signal + nAP*(1 + nAP/nAN)*(medianAn + darkCurrent + readNoise**2)))
    uncertaintyApMag = 1.0857/SNR
    uncertaintySignal = signal/SNR

    print(f'Centroid at: {center}')
    print(f'Signal: {signal:.2f} ± {uncertaintySignal:.2f} ADU')
    print(f'SNR: {SNR:.2f}')
    print(f'Magnitude Instrument: {mInst:.2f} ± {uncertaintyApMag:.2f} mag')
    return mInst

def get_rhohat(RA, dec):
    rhoZ = sin(dec)
    rhoY = sin(RA)*cos(dec)
    rhoX = cos(RA)*cos(dec)

    rhoHatVec = [rhoX, rhoY, rhoZ]
    return np.array(rhoHatVec)

def final_elements(rVec, rDotVec, jDate, newJulian = 0):
    h, a, e, i, longAsc, peri = get_orbital_e(rVec, rDotVec)
    r = mag(rVec)
    k = .01720209895
    truAn = trueanomaly(a, e, h, rVec, rDotVec)
    if(truAn <= 180):
        eccAn = acos((1/e)*(1-(r/a)))
    else:
        eccAn = 2*pi - acos((1/e)*(1-(r/a)))

    meanAn = eccAn - e*sin(eccAn)
    n = k*sqrt(1/a**3)
    perihelionTime = -((meanAn/sqrt(k**2/a**3))-jDate)
    newMeanAn = (newJulian - perihelionTime)*n
    p = jDate - perihelionTime
    print('Orbital Elements of Asteroid')
    print(f"Semimajor axis: {a} AU")
    print(f'Eccentricity: {e}')
    print(f'Inclination: {i*180/pi} degrees')
    print(f'Longitudinal Ascension: {degrees(longAsc)} degrees')
    print(f'Argument of periapsis: {degrees(peri)} degrees')
    print(f'Mean Anomaly: {meanAn*180/pi} degrees at central observation')
    print(f'Mean Anomaly  = {newMeanAn * 180/pi} degrees at JD = {newJulian}')
 
    print(f'E = {eccAn*180/pi}')    
    print(f'n = {n*180/pi} degrees per day')
    print(f'JD of last perihelion passage = {perihelionTime}')
    print(f'P = {p/365.25} years')

    return rVec, rDotVec, h, a, e, i, longAsc, peri, meanAn, perihelionTime

def quadFinding(cos, sin):
    #Finds the cuadrant of a function
    if(abs(cos) > 1 or abs(sin) > 1):
        print("Invalid angle was given")
        return None
    angleC = acos(cos)
    angleS = asin(sin)
    realAngle = 0
    if(cos < 0 and sin < 0):
        realAngle = pi + abs(angleS)
    elif(cos < 0):
        realAngle = angleC
    elif(sin < 0 ):
        realAngle = 2*pi-abs(angleS)
    else:
        realAngle = angleS
    return realAngle * 180/pi

def angleConverter(degrees, minutes, seconds, choiceRad = False, normalize = False):
    # perform angle conversion
    newAngle = copysign(abs(degrees) + minutes/60 + seconds/3600, degrees)
    if(choiceRad):
        newAngle = newAngle* pi/180
        if(normalize):
            while(newAngle > 2* pi):
                newAngle -= 2* pi
            while(newAngle < 0):
                newAngle += 2 * pi 
        return newAngle
              
    if(normalize):
        while(newAngle > 360):
            newAngle -= 360
        while(newAngle < 0):
            newAngle += 360
    # return result
    return (newAngle)

def momentum(rVec, rDotVec):
    return np.around(np.cross(rVec,rDotVec),6)

def centroid(fits_file, target_x, target_y, radius=3, sky_radius=5):
    sky = fits_file[target_y-sky_radius-radius:target_y+sky_radius+radius+1, target_x-sky_radius-radius:target_x+sky_radius+1+radius]
    subImage = fits_file[target_y-radius:target_y+radius+1, target_x-radius:target_x+radius+1]
    #Gets the median skyValue, subtracting away the ring 
    totalSky = np.sum(sky)
    totalAst = np.sum(subImage)
    skyVal = (totalSky-totalAst)/(sky.size-subImage.size)
    adjustedSub = subImage-skyVal
    x_grid, y_grid = np.meshgrid(np.arange(0,adjustedSub.shape[1]), np.arange(0,adjustedSub.shape[0]))
    x_center = np.sum((adjustedSub*x_grid))/np.sum(adjustedSub)
    y_center = np.sum((adjustedSub*y_grid))/np.sum(adjustedSub)
    return x_center+target_x-radius,y_center+target_y-radius

def mag(vector):
    return sqrt(vector[0]**2+vector[1]**2+vector[2]**2)

def semimajor(posVec, velVec): #semimajor axis
    posMag = mag(posVec)
    velMag = mag(velVec)
    return (1/((2/posMag)-velMag**2))

def eccentricity(semimajor, angular): #eccentricity
    magAng = mag(angular)
    return sqrt(1-magAng**2/semimajor)

def inclination(angularM): #inclination
    return atan(sqrt(angularM[0]**2+angularM[1]**2)/angularM[2])

def longacs(angularM, inc): #longitudinal ascention 
    cosVal = (-angularM[1])/(mag(angularM)*sin(inc))
    sinVal = (angularM[0])/(mag(angularM)*sin(inc))
    return quadFinding(cosVal, sinVal)

def trueanomaly(a, e, h, r, rdot): #true anomaly
    sinVal = ((a*(1-e**2))/(e*mag(h)))*(np.dot(r, rdot)/mag(r))
    cosVal = (1/e)*((a*(1-e**2))/mag(r)-1)
    return quadFinding(cosVal, sinVal)

def findU(rVec, longAsc, inc): #U values
    cosVal = (rVec[0]*cos(longAsc)+rVec[1]*sin(longAsc))/mag(rVec)
    sinVal = rVec[2]/(mag(rVec)*sin(inc))
    return quadFinding(cosVal, sinVal)

def arg_peri(U, trueAn):  #Finds the argument of perihilium
    peri = (U-trueAn)%360
    return peri

def get_orbital_e(rVecArr, rDotVec):
    h = momentum(rVecArr, rDotVec)
    a = semimajor(rVecArr, rDotVec)
    e = eccentricity(a, h)
    i = inclination(h)
    longAsc = longacs(h, i)
    trueAn = trueanomaly(a,e,h,rVecArr, rDotVec)
    U = findU(rVecArr, longAsc, i)
    peri = arg_peri(U, trueAn)
    return h, a, e, i, radians(longAsc), radians(peri) #in degrees

def LSPR(filename, centroid):
    data = np.loadtxt(filename, dtype='str')
    xList = [float(data[i, 0]) for i in range(len(data[:, 0]))]
    yList = [float(data[i, 1]) for i in range(len(data[:, 1]))]
    raList = [float(d.split(':')[0])*15 + float(d.split(':')[1])/60*15 + float(d.split(':')[2])/3600*15 for d in data[:, 2]]
    decList = [float(d.split(':')[0]) + float(d.split(':')[1])/60 + float(d.split(':')[2])/3600 for d in data[:, 3]]

    xList = np.array(xList)
    yList = np.array(yList)
    raList = np.array(raList)
    decList = np.array(decList)

    sumX = np.sum(xList)
    sumY = np.sum(yList)
    sumRA = np.sum(raList)
    sumDec = np.sum(decList)

    sumXSquared = np.sum(xList**2)
    sumYSquared = np.sum(yList**2)
    sumXY = np.dot(xList,yList)
    sumRAX = np.dot(raList, xList)
    sumRAY = np.dot(raList, yList)
    sumDecX = np.dot(decList, xList)
    sumDecY = np.dot(decList, yList)

    N = len(xList)

    mat1 = np.array([[N, sumX, sumY], [sumX, sumXSquared, sumXY], [sumY, sumXY, sumYSquared]])
    mat1Inv = np.linalg.inv(mat1)
    raMat = [[sumRA], [sumRAX], [sumRAY]]
    decMat = [[sumDec], [sumDecX],[sumDecY]]

    #the final vectors for the offest
    raResMat = np.dot(mat1Inv, raMat)
    decResMat = np.dot(mat1Inv, decMat)

    #the final RA and Dec
    finalRA = raResMat[0] + raResMat[1]*centroid[0]+ raResMat[2]*centroid[1]
    finalDec = decResMat[0] + decResMat[1]*centroid[0] + decResMat[2]*centroid[1]

    #the uncertainty for RA and Dec
    sumUncertaintyRA = 0
    for i in range(N):
        tempVal = raList[i] - raResMat[0] - raResMat[1]*xList[i] - raResMat[2]*yList[i]
        sumUncertaintyRA += tempVal**2

    sumUncertaintyDec = 0
    for i in range(N):
        tempVal = decList[i] - decResMat[0] - decResMat[1]*xList[i] - decResMat[2]*yList[i]
        sumUncertaintyDec += tempVal**2

    uncertaintyRA = np.around(sqrt(sumUncertaintyRA/(N-3))*3600,2)
    uncertaintyDec = np.around(sqrt(sumUncertaintyDec/(N-3))*3600,2)

    #Converting to the correct format
    final_ra = str(int(finalRA//15)) + ':' + str(int((finalRA/15-finalRA//15)*60)) + ':' + str(np.around(float((((finalRA/15-finalRA//15)*60)-int(((finalRA/15-finalRA//15)*60)))*60),2))
    final_dec = str(int(finalDec)) + ':' + str(int((finalDec-int(finalDec))*60)) + ':' + str(np.around(float(((finalDec-int(finalDec))*60)-int((finalDec-int(finalDec))*60))*60,2))

    print('--'*15)
    print('plate constants')
    print('--'*15)
    print(f'b1: {float(raResMat[0])} deg')
    print(f'b2: {float(decResMat[0])} deg')
    print(f'a11: {float(raResMat[1])} deg/pix')
    print(f'a12: {float(raResMat[2])} deg/pix')
    print(f'a21: {float(decResMat[1])} deg/pix')
    print(f'a22: {float(decResMat[2])} deg/pix')
    print('--'*15)
    print('uncertainty')
    print('--'*15)
    print(f'RA: {uncertaintyRA} arcsec')
    print(f'Dec: {uncertaintyDec} arcsec')
    print('--'*30)
    print('astrometry for')
    print(f'(x,y) = {centroid}')
    print('--'*30)
    print(f'RA = {final_ra}')
    print(f'Dec = {final_dec}')
    return finalRA, finalDec
