# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
from operator import add
from numpy.linalg import inv
import numpy as np

from pyspark import SparkContext


if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  #adding the column of all ones
  yxlines = yxinputFile.map(lambda line: line.split(',')).map(lambda line: map(float,line)).map(lambda line: np.insert(line,1,1))

  #transforming the matrix and multiplying the two matrices
  linregtransf = yxlines.flatMap(lambda linx: [(1,np.multiply((np.matrix(linx[1:len(linx)]).T),np.matrix(linx[1:len(linx)]))),(2,np.multiply(np.matrix(linx[1:len(linx)]).T,linx[0]))])

  #taking the summation
  linregsumm = linregtransf.reduceByKey(add)
  var = linregsumm.collect()

  #taking the product of the inverse of the (X*X.T) and (Y*X.T)
  beta = inv(var[1][1]) * var[0][1]
  print "beta: "
  for coeff in beta:
    print float(coeff)

  sc.stop()
