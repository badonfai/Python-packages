## Copyright (c) 2019, Balint Badonfai
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## * Redistributions of source code must retain the above copyright notice, this
## list of conditions and the following disclaimer.
##
## * Redistributions in binary form must reproduce the above copyright notice,
## this list of conditions and the following disclaimer in the documentation
## and/or other materials provided with the distribution.
##
## * Neither the name of the copyright holder nor the names of its
## contributors may be used to endorse or promote products derived from
## this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
## FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
## OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##
## Written by Balint Badonfai,
## bbadonfai@gmail.com,
## Berlin, Germany

##=======================================================================================
##
## This script of R functions re-implements this Mathematica package:
##
## [1] Anton Antonov, Implementation of one dimensional outlier identifying algorithms in Mathematica,
##     Mathematica package OutlierIdentifiers.m, (2013), MathematicaForPrediction project at GitHub,
##     https://github.com/antononcube/MathematicaForPrediction/blob/master/OutlierIdentifiers.m .
##
##
##=======================================================================================

import numpy as np


#' Hampel identifier parameters.
#' @description Find an Hampel outlier threshold for a data vector.
#' @param data A data vector.
#' @export
def HampelIdentifierParameters(data):
    data = data[~np.isnan(data.astype(float))]
    x0 = np.median(data)
    md = 1.4826 * np.median(np.abs(data-x0))
    return x0-md, x0+md

#' Quartile identifier parameters
#' @description Find an Quartile outlier for a data vector.
#' @param data A data vector.
#' @export
def QuartileIdentifierParameters(data):
    data = data[~np.isnan(data.astype(float))]
    res = np.quantile(data, q=[1/4, 1/2, 3/4])
    xL = res[0]
    x0 = res[1]
    xU = res[2]
    return x0 - (xU - xL), x0 + (xU - xL)

#' SPLUS quartile identifier parameters
#' @description Find an SPLUS Quartile outlier for a data vector.
#' @param data A data vector.
#' @export
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

#' Outlier identifier.
#' @description Find an outlier threshold for a data vector.
#' @param data A data vector.
#' @param identifier An outlier identifier function.
#' @return A numeric vector of outliers.
#' @details The outlier identifier function \code{identifier}
#' should return a list or tuple of two numbers \code{c(lowerThreshold, upperThreshold)}.
#' @export
def OutlierIdentifier(data, identifier):
    if any((isinstance(x, (int, float)) for x in data)):
        raise Exception('The argument data is expected to be a numeric vector')
    lowerThreshold, upperThreshold = identifier(data)
    return data[(data <= lowerThreshold) | (data >= upperThreshold)]

#' Top outlier identifier.
#' @description Find the top outliers for a data vector
#' @param data A data vector.
#' @param identifier An outlier identifier function.
#' @export
def TopOutlierIdentifier(data, identifier):
    lowerThreshold, upperThreshold = identifier(data)
    return data[data >= upperThreshold]

#' Bottom outlier identifier.
#' @description Find the bottom outliers for a data vector.
#' @param data data vector
#' @param identifier An outlier identifier function.
#' @export
def BottomOutlierIdentifier(data, identifier):
    lowerThreshold, upperThreshold = identifier(data)
    return data[data <= lowerThreshold]

#' Outlier positions finder.
#' @description Find the outlier positions in a data vector.
#' @param data A data vector.
#' @param identifier An outlier identifier function.
#' @export
def OutlierPosition(data, identifier=SPLUSQuartileIdentifierParameters):
    lowerThreshold, upperThreshold = identifier(data)
    return np.where((data <= lowerThreshold) | (data >= upperThreshold))[0]

#' Top outlier positions finder.
#' @description Find the top outlier positions in a data vector.
#' @param data A data vector.
#' @param identifier An outlier identifier function.
#' @export
def TopOutlierPosition (data, identifier=SPLUSQuartileIdentifierParameters):
    lowerThreshold, upperThreshold = identifier(data)
    return np.where(data >= upperThreshold)[0]

#' Bottom outlier positions finder.
#' @description Find the bottom outlier positions in a data vector.
#' @param data A data vector.
#' @param identifier An outlier identifier function.
#' @export
def BottomOutlierPosition(data, identifier=SPLUSQuartileIdentifierParameters):
    lowerThreshold, upperThreshold = identifier(data)
    return np.where(data <= lowerThreshold)[0]

