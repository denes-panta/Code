# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gp
import random
import numbers

from shapely.geometry import Point
from pandas.api.types import is_numeric_dtype
from haversine import haversine
from sklearn import cluster as clust

random.seed(1)
pd.options.mode.chained_assignment = None
os.chdir('D:\\Code\\Interviews\\sentience')


# Node to perform Data Engineering
class engineerNode(object):
    
    def __init__(self, csv_name = None, sep = ','):
        # csv_name : name of the input csv file
        # sep : separator used in the csv
        
        self.df = pd.read_csv(csv_name, sep)
        self.df = self.parseData()

        return None
    
    # Function to get the Data from the class
    def getDf(self):
        
        return self.df
    
    # Function to turn the data into the required format
    def parseData(self):
        df = self.df
        
        df['offset'] = df['start_time(YYYYMMddHHmmZ)'].str.slice(12, 17)
    
        df['date_time'] = df['start_time(YYYYMMddHHmmZ)'].str.slice(0, 12)
        df['date_time'] = pd.to_datetime(df['date_time'], format = '%Y%m%d%H%M')
        df['date'] = df['date_time'].dt.date
        df['weekday'] = df['date_time'].dt.weekday
        df['hour'] = df['date_time'].dt.hour
    
        df['duration_h'] = df['duration(ms)'] / (3600 * 60 * 24)

        df.drop(['start_time(YYYYMMddHHmmZ)'], axis = 1, inplace = True)
        df.drop(['duration(ms)'], axis = 1, inplace = True)
        df.drop(['date_time'], axis = 1, inplace = True)
        
        return df
    
    # Function to calculate the distance between each consecutive location
    # Haversine distance measure is used, due to the Earth's curviture
    def calcTimeDistance(self):
        df = self.df
            
        df1 = df.shift(1)[['latitude', 'longitude']]
        df1.iloc[0, : ] = df1.iloc[1, : ]
        df2 = df[['latitude', 'longitude']]
        
        dfM = pd.concat([df1, df2], axis = 1)
        
        dfD = dfM.apply(lambda r: haversine(tuple(r[:2]), tuple(r[2:])), axis = 1)
        dfD = dfD * 1000
        
        self.df['distance'] = dfD
        
        return None

    # Function to get the coordinate columns in radians
    def getRadianCoordDf(self):
        df = self.df[['latitude', 'longitude']]
        
        df.loc[:, 'latitude'] = df['latitude'] * np.pi / 180
        df.loc[:, 'longitude'] = df['longitude'] * np.pi / 180

        return df


# Node to perform analytics on the data
class analyticsNode(object):
    
    def __init__(self, df):
        # df : Input dataframe

        if not isinstance(df, pd.DataFrame):
            return print('df must be a pandas dataframe')
        
        self.df = df
        
        return None
    
    # Function to print a desired descriptive statistic of the dataframe.
    def printStat(self, FUN, col_1, col_2 = None, by = None):
        # FUN : Input dataframe     
        # col_1 : column name
        # col_2 : column name if the FUN reqires two
        # by : column to group by, if any     
        
        df = self.df
        
        if type(FUN) != str:
            return print('FUN needs to be a function name string.')
        elif col_1 not in df.columns:
            return print('col_1 needs to be a column of the df.')
        elif col_2 != None and col_2 not in df.columns:
            return print('col_2 needs to be a column of the df.')
        elif not is_numeric_dtype(df[col_1]):
            return print('col_1 needs to be numeric.')
        elif col_2 != None and not is_numeric_dtype(df[col_2]):
            return print('col_2 needs to be numeric.')
        elif by != None and by not in df.columns:
            return print('by needs to be a column of the df.')
        
        if col_2 == None:
            try:
                if by == None:
                    return eval('df[col_1].' + FUN)()
                else:
                    return eval('df.groupby([by])[col_1].' + FUN)()
                
            except(NameError):
                return print('Data.Frame has no function like: ' + FUN)
                
        else:
            try:
                if by == None:
                    return eval('df[[col_1, col_2]].' + FUN)()
                else:
                    return eval('df.groupby([by])[[col_1, col_2]].' + FUN)()
                
            except(NameError):
                return print('Data.Frame has no function like: ' + FUN)
            
    # Function to show the distribution of a specific column
    def showDist(self, col_1, bins = 30):
        # col_1 : column name
        # bins : number of bins on the histogram
        
        df = self.df
        
        if col_1 not in df.columns:
            return print('col_1 needs to be a column of the df.')
        pltHist = df[col_1].hist(bins = bins)
        plt.show()
        plt.close()
        
        return pltHist
    
    # Function to show the locations plotted on a map
    def showTravelLocations(self):
        df = self.df
        world = gp.read_file(gp.datasets.get_path('naturalearth_lowres'))

        pltMap, ax = plt.subplots(figsize = (14, 8))
        
        lGeometry = [Point(xy) for xy in zip(df['latitude'], df['longitude'])] 
        geo_df = gp.GeoDataFrame(geometry = lGeometry)

        world.plot(ax = ax, color = 'white', edgecolor = 'black')
        geo_df.plot(ax = ax)
        plt.close()
        
        return pltMap
    
    
# Node to check if a person visited an area around a location
class lookupNode(object):
    
    def __init__(self, df):
        # fAvgEarthRad = Average Earth Radius
        # df : Dataframe containing the latitudes and 
        
        self.fAvgEarthRad = 6371.0088
        self.df = df
        
        return None

    # Function to calculate the bounding boxes based on the coordinates
    # Source: http://janmatuschek.de/LatitudeLongitudeBoundingCoordinates
    def calcBondCoord(self, lat, lon, dist):
        # lat : latitude of the location
        # lon : longitude of the location
        # dist : the radius in which we are searching for a point
        
        fDist = dist / 1000
        fAngRad = fDist / self.fAvgEarthRad
        fPiDiv2 = np.pi / 2
        
        fLatRad, fLonRad = map(np.radians, [lat, lon])      
        fLatMin = fLatRad - fAngRad
        fLatMax = fLatRad + fAngRad

        self.fLatRad = fLatRad
        self.fLonRad = fLonRad     
        self.fAngRad = fAngRad

        if fLatMax > fPiDiv2:
            lBoundMin = [fLatMin, -np.pi]
            lBoundMax = [fPiDiv2, np.pi]   

            return 1, lBoundMin, lBoundMax
        
        elif fLatMin < -fPiDiv2:
            lBoundMin = [-fPiDiv2, -np.pi]
            lBoundMax = [fLatMax, np.pi]

            return 1, lBoundMin, lBoundMax
            
        fLonDelt = np.arcsin(np.sin(fAngRad)/np.cos(fLatRad))
        
        fLonMin = fLonRad - fLonDelt
        fLonMax = fLonRad + fLonDelt
            
        if fLonMin < -np.pi:
            lBoundMin_1 = [fLatMin, fLonMin + 2 * np.pi]
            lBoundMax_1 = [fLatMax, np.pi]
            lBoundMin_2 = [fLatMin, -np.pi]
            lBoundMax_2 = [fLatMax, fLonMax]

            return 2, lBoundMin_1, lBoundMax_1, lBoundMin_2, lBoundMax_2 
        
        elif fLonMax > np.pi:
            lBoundMin_1 = [fLatMin, fLonMin]
            lBoundMax_1 = [fLatMax, np.pi]
            lBoundMin_2 = [fLatMin, -np.pi]
            lBoundMax_2 = [fLatMax, fLonMax - 2 * np.pi]
    
            return 2, lBoundMin_1, lBoundMax_1, lBoundMin_2, lBoundMax_2             

        lBoundMin = [fLatMin, fLonMin]
        lBoundMax = [fLatMax, fLonMax]

        return 1, lBoundMin, lBoundMax

    # Function to check if a person has visited an area
    # Haversine distance measure is used, due to the Earth's curviture
    def lookupCoord(self, lat, lon, dist = 3):
        # lat : latitude of the location
        # lon : longitude of the location
        # dist : the radius in which we are searching for a point
        
        if lat < -85.05112878 or lat > 85.05112878:
            return print('Invalid latitude.')
        elif lon < -180 or lon > 180:
            return print('Invalid longitude.')
        elif dist < 0:
            return print('The distance must be positive.')
        
        self.fLatDeg = lat
        self.fLonDeg = lon
        self.fRadius = dist
        
        df = self.df
        lBound = self.calcBondCoord(lat, lon, dist)
        
        fLatRad = self.fLatRad
        fLonRad = self.fLonRad  
        fAngRad = self.fAngRad

        if lBound[0] == 1:
            iRectNum = 1
            lBoundMin = lBound[1]
            lBoundMax = lBound[2]
            
        elif lBound[0] == 2:
            iRectNum = 2            
            lBoundMin_1 = lBound[1]
            lBoundMax_1 = lBound[2]
            lBoundMin_2 = lBound[3]
            lBoundMax_2 = lBound[4]

        if iRectNum == 1:
            df = df[(df['latitude'] >= lBoundMin[0]) & 
                    (df['longitude'] >= lBoundMin[1]) &
                    (df['latitude'] <= lBoundMax[0]) &
                    (df['longitude'] <= lBoundMax[1])]
        
        elif iRectNum == 2:
            df = df[(df['latitude'] >= lBoundMin_1[0]) & 
                    (df['longitude'] >= lBoundMin_1[1]) &
                    (df['latitude'] <= lBoundMax_1[0]) &
                    (df['longitude'] <= lBoundMax_1[1]) |
                    (df['latitude'] >= lBoundMin_2[0]) & 
                    (df['longitude'] >= lBoundMin_2[1]) &
                    (df['latitude'] <= lBoundMax_2[0]) &
                    (df['longitude'] <= lBoundMax_2[1])]

        if len(df) == 0:
            return print('The person has not yet visited the location %.4f, %.4f with radius %.1f m.' 
                         % (self.fLatDeg, self.fLonDeg, self.fRadius))                    
        else:
            return print('The person has already visited the location %.4f, %.4f with radius %.1f m.' 
                         % (self.fLatDeg, self.fLonDeg, self.fRadius))                    
        
        ser = df.apply(
                lambda row: haversine((row[0], row[1]), (fLatRad, fLonRad)),
                axis = 1
                )
        ser = ser[ser <= fAngRad]
            
        if len(ser) == 0:
            return print('The person has not yet visited the location %.4f, %.4f with radius %.1f m.' 
                         % (self.fLatDeg, self.fLonDeg, self.fRadius))                    
        else:
            return print('The person has already visited the location %.4f, %.4f with radius %.1f m.' 
                         % (self.fLatDeg, self.fLonDeg, self.fRadius))                    
                
        
# Node to detect Workplace and Home of a person        
class detectionNode(object):
    
    def __init__(self, df):
        # df : Dataframe containing the Data from the engineer Node 

        if not isinstance(df, pd.DataFrame):
            return print('df must be a pandas dataframe')

        self.df = df
        
        return None

    #Function to create summary dataframe for the clusters       
    def createSummary(self, eps = 0.0004):
        # eps : DBSCAN parameter - The maximum distance between two samples,
        # for one to be considered as in the neighborhood of the other.
        
        if not isinstance(eps, numbers.Number):
            return print('eps must be a numerical value')
        
        df = self.df
        
        # Using DBSCAN algorithm, because I need to know which locations have been
        # visited the most, and I need the stand-alone locations (noise) filtered out
        
        # The distance measure, is the default euclidian distance,
        # because we only care about locations that are a few meters away
        # from each other.
        # In this case we can ignore the planet's curviture
        
        model = clust.DBSCAN(eps = eps).fit(df[['latitude', 'longitude']])
        df.loc[:, 'cluster'] = model.labels_.tolist()
        
        dfS = pd.DataFrame()
        
        dfTemp = df[['duration_h', 'cluster']].groupby('cluster').max()
        dfS['max_duration_h'] = dfTemp['duration_h']
        
        dfTemp = df[['hour', 'cluster']].groupby('cluster').count()
        dfS['count_places'] = dfTemp['hour']
        
        dfTemp = df[['hour', 'cluster']].groupby('cluster').min()
        dfS['min_hour'] = dfTemp['hour']
        
        dfTemp = df[['hour', 'cluster']].groupby('cluster').max()
        dfS['max_hour'] = dfTemp['hour']
        
        dfTemp = df[['hour', 'cluster']].groupby('cluster').sum()
        dfS['sum_duration_h'] = dfTemp['hour']
        
        dfTemp = df[['weekday', 'cluster']].groupby('cluster').max()
        dfS['max_weekday'] = dfTemp['weekday']
        
        dfTemp = df[['latitude', 'cluster']].groupby('cluster').median()
        dfS['latitude'] = dfTemp['latitude']
        
        dfTemp = df[['longitude', 'cluster']].groupby('cluster').median()
        dfS['longitude'] = dfTemp['longitude']

        # Remove the unclustered locations
        dfS = dfS[dfS.index != -1]
    
        self.dfS = dfS
        
        return None
    
    # Function to get the locations of a person's homes
    def getHome(self):
        dfS = self.dfS
        
        # Home is where a person regularly sleeps.
        # This means that for a person with a day job would not spend the nights at home
        # and leave for work in the mornings.
        # This would show in the data:
        # - A persons phone would spend 6 to 8 hours motionless,
        # - Since the nights are spent at home, the min_hour should be close to 0
        # - Since the nights are spent at home, the max_hour should be close to 24
        # - Since it is highly unlikely for a normal person to work on weekends,
        #   only for a home, the max weekday should be 6 = Sunday
        
        dfHome = dfS[(dfS['max_weekday'] == 6) & 
                     (dfS['min_hour'] < 4) & 
                     (dfS['max_hour'] > 20)]
        dfHome['type'] = 'secondary'
        
        # The max_duration_h should be the highest for a person's main home
        # - Additionally this would mean that the total number of hours should also be the highest
    
        fSumDurMax = dfHome['sum_duration_h'].max()
        dfHome.loc[fSumDurMax == dfHome['sum_duration_h'], ['type']] = 'main'
        dfHome = dfHome[['latitude', 'longitude', 'type']]
        
        if len(dfHome) == 0:
            print('The person does not seem to have a stable home.')
        else:
            print('The person has the following homes:')   
        print(dfHome)
        
        self.dfHome = dfHome
    
        return dfHome
        
    # Function to get the locations of a person's workplaces    
    def getWork(self):
        dfS = self.dfS
        
        # Work is where a person with a day job spends his workdays.
        # We assume that a person doesn't work on weekends and works Monday to Friday
        # This would show in the data:
        # - min_hour would be between 6 and 10
        # - max_hour would be between 16 and 20
        # - max_weekday would be 4 = Friday
        
        dfWork = dfS[(dfS['max_weekday'] == 4) & 
                     (dfS['min_hour'] < 11) & 
                     (dfS['min_hour'] > 5) & 
                     (dfS['max_hour'] < 21) & 
                     (dfS['max_hour'] > 15)]
        
        dfWork = dfWork[['latitude', 'longitude']]

        if len(dfWork) == 0:
            print('The person does not seem to have a stable workplace.')
        else:
            print('The person has the following workplaces:')
        
        self.dfWork = dfWork
        print(dfWork)
        
        return dfWork
    
    # Function to show the work or home locations
    def printLocationMap(self, work_home = 'home'):
        # work_home : wheter we want to print the work or home locations
        
        if work_home == 'home':
            df = self.dfHome
        elif work_home == 'work':
            df = self.dfWork
        else:
            return print('work_home must be "home" or "work"')
        
        world = gp.read_file(gp.datasets.get_path('naturalearth_lowres'))

        pltMap, ax = plt.subplots(figsize = (14, 8))
        
        lGeometry = [Point(xy) for xy in zip(df['latitude'], df['longitude'])] 
        geo_df = gp.GeoDataFrame(geometry = lGeometry)

        world.plot(ax = ax, color = 'white', edgecolor = 'black')
        geo_df.plot(ax = ax)
        plt.show()
        plt.close()
        
        return None
        
    
if __name__ == "__main__":  
    
    # Question 1A - Data Parsing
    oEngNode = engineerNode('person_1.csv', sep = ';')
    oEngNode.calcTimeDistance()
    
    # Question 1A - Summary statistics
    oAnaNode = analyticsNode(oEngNode.getDf())
    # Median distance travelled by date
    print('Median of the distances by date:')
    print(oAnaNode.printStat(FUN = 'median', col_1 = 'distance', by = 'date'))
    # Mean duration stayed stationary by date
    print('Mean duration stayed at location by date:')
    print(oAnaNode.printStat(FUN = 'mean', col_1 = 'duration_h', by = 'date'))
    # Kurtosis of the distance travelled
    print('Kurtosis of the distances travelled:')
    print(oAnaNode.printStat(FUN = 'kurtosis', col_1 = 'distance'))
    # Description of the duration stayed stationary
    print('Descriptive statistics of the duration stayed at a place:')
    print(oAnaNode.printStat(FUN = 'describe', col_1 = 'duration_h'))
    # Correlation between distance travelled and time stayed stationary
    print('Descriptive statistics of the duration stayed at a place:')
    print(oAnaNode.printStat(FUN = 'corr', col_1 = 'duration_h', col_2 = 'distance'))
    # Distribution of duration
    print('Histogram of the duration_h feature:')
    oAnaNode.showDist(col_1 = 'duration_h', bins = 50)
    # Distribution of places visited by day (0 = Monday, 6 = Sunday)
    print('Histogram of the weekdays variable:')
    oAnaNode.showDist(col_1 = 'weekday', bins = 50)
    # Distribution of places visited by hour
    print('Histogram of the hour variable:')
    oAnaNode.showDist(col_1 = 'hour', bins = 50)
    # Visualisze the coordinates on a map
    print('Locations visualised on a world map:')    
    oAnaNode.showTravelLocations()
    
    # Question 2 - Lookup
    oLookUpNode = lookupNode(oEngNode.getRadianCoordDf())
    # Coordinate in the person_1.csv file with 1 meter accuracy
    oLookUpNode.lookupCoord(51.1716, 4.34697, 1)
    # Coordinate not in the person_1.csv file with 1 meter accuracy
    oLookUpNode.lookupCoord(80, 180, 1)
    # Approximate coordinate not in the person_1.csv file with 20 km accuracy
    oLookUpNode.lookupCoord(51, 4, 20 * 1000)
    
    # Question 3 - Detect Home / Work
    oDetNode = detectionNode(oEngNode.getDf())
    oDetNode.createSummary(eps = 0.0005)
    # Get home locations
    oDetNode.getHome()
    # Show home locations
    oDetNode.printLocationMap('home')
    # Get work locations
    oDetNode.getWork()
    # Show work locations
    plt.show(oDetNode.printLocationMap('work'))
    
    