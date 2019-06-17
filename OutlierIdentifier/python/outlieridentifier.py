import numpy as np


def HampelIdentifierParameters(data):
    data = data[~np.isnan(data.astype(float))]
    x0 = np.median(data)
    md = 1.4826 * np.median(np.abs(data-x0))
    return x0-md, x0+md


def QuartileIdentifierParameters(data):
    data = data[~np.isnan(data.astype(float))]
    res = np.quantile(data, q=[1/4, 1/2, 3/4])
    xL = res[0]
    x0 = res[1]
    xU = res[2]
    return x0 - (xU - xL), x0 + (xU - xL)


def SPLUSQuartileIdentifierParameters(data):
    data = data[~np.isnan(data.astype(float))]
    if len(data) <= 4:
        xL = np.min(data)
        xU = np.max(data)
    else:
        res = np.quantile(data, q=[1/4, 3/4])
        xL = res[0]
        xU = res[1]

    return xL - 1.5*(xU-xL), xU + 1.5*(xU-xL)


def OutlierIdentifier(data, identifier):
    if any((isinstance(x, (int, float)) for x in data)):
        raise Exception('The argument data is expected to be a numeric vector')
    lowerThreshold, upperThreshold = identifier(data)
    return data[(data <= lowerThreshold) | (data >= upperThreshold)]


def TopOutlierIdentifier(data, identifier):
    lowerThreshold, upperThreshold = identifier(data)
    return data[data >= upperThreshold]


def BottomOutlierIdentifier(data, identifier):
    lowerThreshold, upperThreshold = identifier(data)
    return data[data <= lowerThreshold]


def OutlierPosition(data, identifier=SPLUSQuartileIdentifierParameters):
    lowerThreshold, upperThreshold = identifier(data)
    return np.where((data <= lowerThreshold) | (data >= upperThreshold))[0]


def TopOutlierPosition (data, identifier=SPLUSQuartileIdentifierParameters):
    lowerThreshold, upperThreshold = identifier(data)
    return np.where(data >= upperThreshold)[0]


def BottomOutlierPosition(data, identifier=SPLUSQuartileIdentifierParameters):
    lowerThreshold, upperThreshold = identifier(data)
    return np.where(data <= lowerThreshold)[0]

