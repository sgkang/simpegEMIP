import numpy as np

def mapDat(dat, cutoff, stretch=1):
    datM = np.empty_like(dat)
    datM.fill(np.nan)

    indLin = np.abs(dat) <= cutoff
    datM[indLin] = dat[indLin]/cutoff

    indPos = dat > cutoff
    datM[indPos] = np.log10(dat[indPos])-np.log10(cutoff)+1

    indNeg = dat < -cutoff
    datM[indNeg] = -np.log10(-dat[indNeg])+np.log10(cutoff)-1

    datMS = datM.copy()
    datMS[indPos] = stretch*datM[indPos] - stretch + 1
    datMS[indNeg] = stretch*datM[indNeg] + stretch - 1

    maxTick = np.ceil(datM.max())
    minTick = np.floor(datM.min())

    pTicks = np.r_[1:maxTick]
    pTicks = stretch*pTicks + 1
    nTicks = np.r_[1:-minTick]
    nTicks = -(stretch*nTicks+1)

    ticks = np.sort(np.r_[nTicks, -1, 0, 1, pTicks])

    nTickVals = np.r_[np.log10(cutoff)-minTick-1:np.log10(cutoff)-1:-1]
    pTickVals = np.r_[np.log10(cutoff):np.log10(cutoff)+maxTick]

    tickLabels = []
    for tck in nTickVals:
        tckStr = "$-10^{"+str(int(tck))+"}$"
        tickLabels.append(tckStr)
    tickLabels.append("$0$    ")
    for tck in pTickVals:
        tckStr = "$10^{"+str(int(tck))+"}$"
        tickLabels.append(tckStr)

    return datMS, ticks, tickLabels