import pandas
from multiprocessing import Pool, freeze_support
from pylspm import PyLSpm
from results import PyLSpmHTML
from boot import PyLSboot
import numpy as np
from scipy.stats import norm
from numpy import inf
import pandas as pd
import copy


if __name__ == '__main__':
    freeze_support()

    boot = 2
    nrboot = 10
    method = 'bca'
    data = 'dados2.csv'
    cores = 8
    scheme = 'path'
    regression = 'ols'

    data_ = pandas.read_csv(data)

    if (boot==1):
        tese = PyLSboot(nrboot, cores, data, 'lvmodel.csv', 'mvmodel.csv', scheme, regression, 0, 100)
        resultados = tese.boot()

        current = list(filter(None.__ne__, resultados))
        current = np.sort(current, axis=0)

        default = PyLSpm(data, 'lvmodel.csv', 'mvmodel.csv', 0, 100).path_matrix.values

        if (method == 'percentile'):
            for i in range(len(current[0])):
                print(i)
                current_ = [j[i] for j in current]
                print('MEAN')
                print( np.round(np.mean(current_, axis=0),4) )
                print('STD')
                print( np.round(np.std(current_, axis=0, ddof=1),4) )
                print('CI 2.5')
                print ( np.round(np.percentile(current_, 2.5, axis=0),4) )
                print('CI 97.5')
                print ( np.round(np.percentile(current_, 97.5, axis=0),4) )

        elif (method == 'bca'):
            for i in range(len(current[0])):
                current_ = [j[i] for j in current]

                alpha=0.05
                if np.iterable(alpha):
                    alphas = np.array(alpha)
                else:
                    alphas = np.array([alpha/2,1-alpha/2])

                # bias
                z0 = norm.ppf( ( np.sum(current_ < default, axis=0)  ) / len(current_) )
                zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

                # acceleration and jackknife
                jstat = PyLSboot(len(data_), cores, data_, 'lvmodel.csv', 'mvmodel.csv', scheme, regression, 0, 100)
                jstat = jstat.jk()
                jstat = list(filter(None.__ne__, jstat))
                
                jmean = np.mean(jstat,axis=0)
                a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**(3/2) )
                zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

                avals = norm.cdf(z0 + zs/(1-a*zs))

                nvals = np.round((len(current_)-1)*avals)
                nvals = np.nan_to_num(nvals).astype('int')

                low_conf = np.zeros(shape=(len(current_[0]), len(current_[0])))
                high_conf = np.zeros(shape=(len(current_[0]), len(current_[0])))

                for i in range(len(current_[0])):
                    for j in range(len(current_[0])):
                        low_conf[i][j] = (current_[nvals[0][i][j]][i][j])

                for i in range(len(*current[0])):
                    for j in range(len(*current[0])):
                        high_conf[i][j] = (current_[nvals[1][i][j]][i][j])

                print('MEAN')
                print( np.round(np.mean(current_, axis=0),4) )
                print('CI LOW')
                print(avals[0])
                print(low_conf)
                print('CI HIGH')
                print(avals[1])
                print(high_conf)

    elif (boot==2):

        def isNaN(num):
            return num != num

        # Blindfolding
        data = pd.read_csv('dados3.csv')

        # observation/distance must not be interger
        distance = 7

        Q2 = pd.DataFrame(0, index=range(len(data.columns)), columns=range(distance))
        Q2.index = data.columns.values
        mean = pd.DataFrame.mean(data)

        for dist in range(distance):
            dataBlind = copy.deepcopy(data)
            rodada=1
            count = distance-dist-1
            for j in range(len(data.columns)):
                for i in range(len(data)):
                    count+=1
                    if count==distance:
                        dataBlind.ix[i,j]=np.nan
                        count=0

            for j in range(len(data.columns)):
                for i in range(len(data)):
                    if (isNaN(dataBlind.ix[i,j])):
                        dataBlind.ix[i,j] = mean[j]
            
            rodada=rodada+1

            plsRound = PyLSpm(dataBlind, 'lvmodel.csv', 'mvmodel.csv', scheme, regression, 0, 100)
            predictedRound = plsRound.predict()

            sse = pd.DataFrame.sum((data-predictedRound)**2)
            sso = pd.DataFrame.sum((data-mean)**2)
            Q2_=1-(sse/sso)
            Q2[dist]=Q2_
        Q2 = pd.DataFrame.mean(Q2, axis=1)
        print(Q2)
        
    elif (boot==0):
        tese = PyLSpm(data, 'lvmodel.csv', 'mvmodel.csv', scheme, regression, 0, 100)
        imprime = PyLSpmHTML(tese)
        imprime.generate()


        