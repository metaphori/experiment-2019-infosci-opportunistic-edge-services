import numpy as np
import xarray as xr
import re

def distance(val, ref):
    return abs(ref - val)
vectDistance = np.vectorize(distance)

def getClosest(sortedMatrix, column, val):
    while len(sortedMatrix) > 3:
        half = int(len(sortedMatrix) / 2)
        sortedMatrix = sortedMatrix[-half - 1:] if sortedMatrix[half, column] < val else sortedMatrix[: half + 1]
    if len(sortedMatrix) == 1:
        result = sortedMatrix[0].copy()
        result[column] = val
        return result
    else:
        safecopy = sortedMatrix.copy()
        safecopy[:, column] = vectDistance(safecopy[:, column], val)
        minidx = np.argmin(safecopy[:, column])
        safecopy = safecopy[minidx, :].A1
        safecopy[column] = val
        return safecopy

def convert(column, samples, matrix):
    return np.matrix([getClosest(matrix, column, t) for t in samples])

def valueOrEmptySet(k, d):
    return (d[k] if isinstance(d[k], set) else {d[k]}) if k in d else set()

def mergeDicts(d1, d2):
    """
    Creates a new dictionary whose keys are the union of the keys of two
    dictionaries, and whose values are the union of values.

    Parameters
    ----------
    d1: dict
        dictionary whose values are sets
    d2: dict
        dictionary whose values are sets

    Returns
    -------
    dict
        A dict whose keys are the union of the keys of two dictionaries,
    and whose values are the union of values

    """
    res = {}
    for k in d1.keys() | d2.keys():
        res[k] = valueOrEmptySet(k, d1) | valueOrEmptySet(k, d2)
    return res

def extractCoordinates(filename):
    """
    Scans the header of an Alchemist file in search of the variables.

    Parameters
    ----------
    filename : str
        path to the target file
    mergewith : dict
        a dictionary whose dimensions will be merged with the returned one

    Returns
    -------
    dict
        A dictionary whose keys are strings (coordinate name) and values are
        lists (set of variable values)

    """
    with open(filename, 'r') as file:
        regex = re.compile(' (?P<varName>[a-zA-Z]+) = (?P<varValue>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),?')
        dataBegin = re.compile('\d')
        for line in file:
            match = regex.findall(line)
            if match:
                return {var : float(value) for var, value in match}
            elif dataBegin.match(line[0]):
                return {}

def extractVariableNames(filename):
    """
    Gets the variable names from the Alchemist data files header.

    Parameters
    ----------
    filename : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    with open(filename, 'r') as file:
        dataBegin = re.compile('\d')
        lastHeaderLine = ''
        for line in file:
            if dataBegin.match(line[0]):
                break
            else:
                lastHeaderLine = line
        if lastHeaderLine:
            regex = re.compile(' (?P<varName>\S+)')
            return regex.findall(lastHeaderLine)
        return []

def openCsv(path):
    """
    Converts an Alchemist export file into a list of lists representing the matrix of values.

    Parameters
    ----------
    path : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    regex = re.compile('\d')
    with open(path, 'r') as file:
        lines = filter(lambda x: regex.match(x[0]), file.readlines())
        return [[float(x) for x in line.split()] for line in lines]

if __name__ == '__main__':
    # CONFIGURE SCRIPT
    directory = 'data'
    pickleOutput = 'data_summary'
    experiments = ['cloud', 'edge', 'reference']
    floatPrecision = '{: 0.2f}'
    seedVars = ['seed']
    timeSamples = 500
    minTime = 0
    maxTime = 3600
    timeColumnName = 'time'
    logarithmicTime = False
    
    # Setup libraries
    np.set_printoptions(formatter={'float': floatPrecision.format})
    # Read the last time the data was processed, reprocess only if new data exists, otherwise just load
    import pickle
    import os
    newestFileTime = max(os.path.getmtime(directory + '/' + file) for file in os.listdir(directory))
    try:
        lastTimeProcessed = pickle.load(open('timeprocessed', 'rb'))
    except:
        lastTimeProcessed = -1
    shouldRecompute = False #newestFileTime != lastTimeProcessed
    if not shouldRecompute:
        try:
            means = pickle.load(open(pickleOutput + '_mean', 'rb'))
            stdevs = pickle.load(open(pickleOutput + '_std', 'rb'))
        except:
            shouldRecompute = True
    if shouldRecompute:
        timefun = np.logspace if logarithmicTime else np.linspace
        means = {}
        stdevs = {}
        for experiment in experiments:
            # Collect all files for the experiment of interest
            import fnmatch
            allfiles = filter(lambda file: fnmatch.fnmatch(file, experiment + '_*.txt'), os.listdir(directory))
            allfiles = [directory + '/' + name for name in allfiles]
            allfiles.sort()
            # From the file name, extract the independent variables
            dimensions = {}
            for file in allfiles:
                dimensions = mergeDicts(dimensions, extractCoordinates(file))
            dimensions = {k: sorted(v) for k, v in dimensions.items()}
            # Add time to the independent variables
            dimensions[timeColumnName] = range(0, timeSamples)
            # Compute the matrix shape
            shape = tuple(len(v) for k, v in dimensions.items())
            # Prepare the Dataset
            dataset = xr.Dataset()
            for k, v in dimensions.items():
                dataset.coords[k] = v
            varNames = extractVariableNames(allfiles[0])
            for v in varNames:
                if v != timeColumnName:
                    novals = np.ndarray(shape)
                    novals.fill(float('nan'))
                    dataset[v] = (dimensions.keys(), novals)
            # Compute maximum and minimum time, create the resample
            timeColumn = varNames.index(timeColumnName)
            allData = { file: np.matrix(openCsv(file)) for file in allfiles }
            computeMin = minTime is None
            computeMax = maxTime is None
            if computeMax:
                maxTime = float('-inf')
                for data in allData.values():
                    maxTime = max(maxTime, data[-1, timeColumn])
            if computeMin:
                minTime = float('inf')
                for data in allData.values():
                    minTime = min(minTime, data[0, timeColumn])
            timeline = timefun(minTime, maxTime, timeSamples)
            # Resample
            for file in allData:
                allData[file] = convert(timeColumn, timeline, allData[file])
            # Populate the dataset
            for file, data in allData.items():
                dataset[timeColumnName] = timeline
                for idx, v in enumerate(varNames):
                    if v != timeColumnName:
                        darray = dataset[v]
                        experimentVars = extractCoordinates(file)
                        darray.loc[experimentVars] = data[:, idx].A1
            # Fold the dataset along the seed variables, producing the mean and stdev datasets
            means[experiment] = dataset.mean(seedVars)
            stdevs[experiment] = dataset.std(seedVars)
        # Save the datasets
        pickle.dump(means, open(pickleOutput + '_mean', 'wb'), protocol=-1)
        pickle.dump(stdevs, open(pickleOutput + '_std', 'wb'), protocol=-1)
        pickle.dump(newestFileTime, open('timeprocessed', 'wb'))

    # Prepare the charting system
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmx
    import matplotlib.patches as patches
    figure_size=(6, 3.5)
    matplotlib.rcParams.update({'axes.titlesize': 14})
    matplotlib.rcParams.update({'axes.labelsize': 13})
#    colormap = cmx.gist_rainbow
    colormap = cmx.viridis
    
    # Edge delay and size with range and ap-count
    packet_size = 'packet_size[Mean]'
    delay = 'mean_delay[Mean]'
    x = means['edge']['range']
    for var in [packet_size, delay]:
        merge_data = means['edge'].mean('time').mean('bandwidth')
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Wireless range (m)")
        ax.set_ylabel({
            packet_size: 'Packet size (Bytes)',
            delay: 'Mean packet delay (s)'
        }[var])
        ax.set_xscale('log')
        ax.set_title('Sensitivity on range and number of edge nodes')
        index = 0
        for ap_count in merge_data['accessPointCount']:
            y = merge_data.data_vars[var].sel({'accessPointCount': ap_count})
            ax.plot(x, y, label=str(int(ap_count)), color=colormap(index / 9))
            index = index + 1
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1])
        ax.add_patch(patches.Rectangle((46 / 3, 0), 92 - 46 / 3, ax.get_ylim()[1], alpha=0.2, color='black'))
        ax.text(x=20, y=ax.get_ylim()[1]*0.9, color='black', s='typical wifi range')
        ax.legend(ncol=5)
        fig.tight_layout()
        fig.savefig(ax.get_title()+ax.get_ylabel()+'.pdf')
    
    # Edge delay and size with bandwidth and ap-count
    x = means['edge']['bandwidth']
    for var in [packet_size, delay]:
        merge_data = means['edge'].mean('time').mean('range')
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Bandwidth (B/s)")
        ax.set_ylabel({
            packet_size: 'Packet size (Bytes)',
            delay: 'Mean packet delay (s)'
        }[var])
        ax.set_xscale('log')
        ax.set_title('Sensitivity on bandwidth and number of edge nodes')
        index = 0
        for ap_count in merge_data['accessPointCount']:
            y = merge_data.data_vars[var].sel({'accessPointCount': ap_count})
            ax.plot(x, y, label=str(int(ap_count)), color=colormap(index / 9))
            index = index + 1
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1])
        ax.add_patch(patches.Rectangle((5.5e6 / 8, 0), 100e6/8 - 5.5e6 / 8, ax.get_ylim()[1], alpha=0.2, color='black'))
        ax.text(x=10e5, y=ax.get_ylim()[1]*0.4, color='black', s='typical\nwifi\nbandwidth')
        ax.legend(ncol=3)
        fig.tight_layout()
        fig.savefig(ax.get_title()+ax.get_ylabel()+'.pdf')
    
    # Cloud delay and size with bandwidth and range
    x = means['cloud']['bandwidth']
    mobile_technologies = {
        "EDGE": [0.1e6 / 8, 0.3e6 / 8],
        "HSPA": [1.5e6 / 8, 7.2e6 / 8],
        "HSPA+": [4e6 / 8, 21e6 / 8],
        "LTE-AC6": [24e6 / 8, 300e6 / 8]
    }
    for var in [packet_size, delay]:
        merge_data = means['cloud'].mean('time').mean('propagationDelay')
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Bandwidth (B/s)")
        ax.set_ylabel({
            packet_size: 'Packet size (Bytes)',
            delay: 'Mean packet delay (s)'
        }[var])
        ax.set_xscale('log')
        ax.set_title('Sensitivity on bandwidth and computation effort')
        index = 0
        for _range in merge_data['range']:
            y = merge_data.data_vars[var].sel({'range': _range})
            ax.plot(x, y, label=str(int(_range)), color=colormap(index / 9))
            index = index + 1
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1])
        for mobile_name, mobile_perf in mobile_technologies.items():
            bw_min = mobile_perf[0] / 2
            bw_max = mobile_perf[1] / 2
            ax.add_patch(patches.Rectangle((bw_min, 0), bw_max-bw_min, ax.get_ylim()[1], alpha=0.2, color='black'))
            ax.text(x=bw_min*1.1, y=ax.get_ylim()[1]*0.45, color='black', s='typical\n' + mobile_name + ' \nbandwidth')
        ax.legend(ncol=3)
        fig.tight_layout()
        fig.savefig(ax.get_title()+ax.get_ylabel()+'.pdf')
    
    # Cloud delay and size with propagation delay and range
    x = means['cloud']['propagationDelay']
    for var in [packet_size, delay]:
        merge_data = means['cloud'].mean('time').mean('bandwidth')
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Propagation delay (s)")
        ax.set_ylabel({
            packet_size: 'Packet size (Bytes)',
            delay: 'Mean packet delay (s)'
        }[var])
        ax.set_xscale('log')
        ax.set_title('Sensitivity on propagation delay and computation effort')
        index = 0
        for _range in merge_data['range']:
            y = merge_data.data_vars[var].sel({'range': _range})
            ax.plot(x, y, label=str(int(_range)), color=colormap(index / 9))
            index = index + 1
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1])
        ax.add_patch(patches.Rectangle((50e-3, 0), 450e-3, ax.get_ylim()[1], alpha=0.2, color='black'))
        ax.text(x=1e-1, y=ax.get_ylim()[1]*0.7, color='black', s='typical\ndelay')
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(ax.get_title()+ax.get_ylabel()+'.pdf')
    
    # Comparison
    edge_sparse = means['edge'].sel({
            'range': 100.0,
            'bandwidth': means['edge']['bandwidth'][6],
            'accessPointCount': means['edge']['accessPointCount'][6]})
    edge_dense = means['edge'].sel({
            'range': means['edge']['range'][3],
            'bandwidth': means['edge']['bandwidth'][6],
            'accessPointCount': means['edge']['accessPointCount'][8]})
    cloud_2g = means['cloud'].sel({
            'range': means['cloud']['range'][6],
            'bandwidth': means['cloud']['bandwidth'][3],
            'propagationDelay': means['cloud']['propagationDelay'][5]})
    cloud_3g = means['cloud'].sel({
            'range': means['cloud']['range'][6],
            'bandwidth': means['cloud']['bandwidth'][4],
            'propagationDelay': means['cloud']['propagationDelay'][4]})
    cloud_4g = means['cloud'].sel({
            'range': means['cloud']['range'][6],
            'bandwidth': means['cloud']['bandwidth'][4],
            'propagationDelay': means['cloud']['propagationDelay'][3]})
    fig = plt.figure(figsize = figure_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel('Mean packet delay (s)')
    ax.set_title('Packet delay: edge/cloud comparison')
#    ax.set_yscale('log')
    ax.plot(edge_sparse['time'], edge_sparse.data_vars['mean_delay[Mean]'], label='edge-sparse')
    ax.plot(edge_sparse['time'], edge_dense.data_vars['mean_delay[Mean]'], label='edge-dense')
    ax.plot(edge_sparse['time'], cloud_2g.data_vars['mean_delay[Mean]'], label='cloud-2G')
    ax.plot(edge_sparse['time'], cloud_3g.data_vars['mean_delay[Mean]'], label='cloud-3G')
    ax.plot(edge_sparse['time'], cloud_4g.data_vars['mean_delay[Mean]'], label='cloud-LTE')
    ax.legend(ncol=2, bbox_to_anchor=(0.37, 0.58))
    fig.savefig('edge-cloud.pdf')
    
    
    #means['edge'].sel(bandwidth=1e7).mean('time').data_vars['packet_size[Mean]']

    if False:
        means['vienna']['CurrentSpeed[Sum]'] = means['vienna']['CurrentSpeed[Sum]'] / 1497
        for datavar in means['vienna'].data_vars:
            fig = plt.figure(figsize = figure_size)
            ax = fig.add_subplot(1, 1, 1)
            matrix = means['vienna'][datavar]
            titles = {
                '@following[Sum]': 'Users being steered',
                '@overcrowded[Sum]': 'Users in dangerous areas',
                '@risk[Sum]': 'Users in moderate risk areas',
                '@warning[Sum]': 'Users warned by the application',
                '@direction[Sum]': 'Users receiving steering information',
                '@crowd[Sum]': 'Users receiving a warning alert with steering information',
                'CurrentSpeed[Sum]': 'Mean walking speed',
                'DistanceTraveled[Sum]': 'Mean walked distance',
                'OverallDistanceTraveled[Sum]': 'Mean total walked distance'
            }
            title = titles[datavar]
            ax.set_title(title)
            x = matrix.coords[timeColumnName] / 60 
            maxy = float('-inf')
            miny = float('inf')
            for dimension in matrix.coords:
                if dimension != timeColumnName:
                    for index, dimval in enumerate(matrix.coords[dimension]):
                        y = matrix.loc[{dimension: dimval}]
                        maxy = max(maxy, y.where(getattr(y, timeColumnName) > 20 * 60).max() * 1.1)
                        miny = min(miny, y.min() * 0.9)
                        ax.set_ylabel('speed (m/s)' if 'Speed' in datavar else 'distance traveled (m)' if 'Distance' in datavar else 'people')
                        ax.set_xlabel('time (simulated minutes)')
                        color = colormap(index / len(matrix.coords[dimension]))
                        ax.plot(x, y, color = color, label = 'p = ' + "{0:.1f}".format(float(dimval)))
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(miny, maxy)
            ax.legend(shadow=True, bbox_to_anchor=[1, 1.05])
            fig.savefig(title.replace(' ', '_') + '.pdf', bbox_inches=matplotlib.transforms.Bbox([[0, -0.1], [6.6, 2.95]]))
       
        # Aggregate chart: average of users in danger and their speed with probability of following the advice
        data = means['vienna']['@overcrowded[Sum]'].mean('time')
        dataerr = means['vienna']['@overcrowded[Sum]'].std('time')
        dataspeed = means['vienna']['CurrentSpeed[Sum]'].mean('time')
        dataspeederr = means['vienna']['CurrentSpeed[Sum]'].std('time')
        x = data.coords['pFollows']
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax2 = ax.twinx()
        title = 'Cost-benefit: average mean speed and people in danger'
        ax.set_title(title)
        errline = ax.plot(x, data + dataerr, color='black', linestyle='--', linewidth=1, label='± standard deviation')
        ax.plot(x, data - dataerr, color='black', linestyle='--', linewidth=1)
        print(dataspeederr)
        ax2.plot(x, dataspeed + dataspeederr, color='red', linestyle='--', linewidth=1)
        ax2.plot(x, dataspeed - dataspeederr, color='red', linestyle='--', linewidth=1)
        line1 = ax.plot(x, data, color='black', label='People in danger')
        line2 = ax2.plot(x, dataspeed, color = 'red', label='Mean people speed')
        ax.set_ylabel('People')
        ax.set_xlabel('Probability of users following the app advice')
        ax2.set_ylabel('Mean people speed (m/s)')
        ax2.set_ylim(0, (dataspeed + dataspeederr).max() * 1.1)
        ax.set_ylim(0, (data + dataerr).max() * 1.1)
        ax.set_xlim(0, 1)
        lines = line1 + errline + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, shadow=False, frameon=False, ncol=2, bbox_to_anchor=[0.05, 0.75])
        fig.savefig('summary.pdf', bbox_inches=matplotlib.transforms.Bbox([[0.25, -0.05], [5.95, 2.9]]))

